from __future__ import annotations

import json
import os
import tempfile
import base64
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
import string
import re


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _slugify_name(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip()).strip("-_")
    return token[:48] or "project"


def _rand_suffix(k: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_path = Path(tf.name)
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tf:
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_path = Path(tf.name)
    os.replace(tmp_path, path)


class ProjectStore:
    """
    Local on-disk project storage (Phase 1):
    - current snapshots: canvas.json / graph.json / runtime.json
    - project metadata: meta.json
    - append-only WAL: wal/ops.jsonl
    """

    SCHEMA_VERSION = 2

    def __init__(self, root_dir: Optional[Path | str] = None) -> None:
        if root_dir is None:
            env_dir = os.getenv("GRAPH_PROJECTS_DIR", "").strip()
            if env_dir:
                p = Path(env_dir)
                self.root = p if p.is_absolute() else (Path(__file__).resolve().parents[1] / p)
            else:
                self.root = Path(__file__).resolve().parents[1] / "data" / "projects"
        else:
            self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        return self.root / project_id

    def _meta_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "meta.json"

    def _current_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "current"

    def _canvas_path(self, project_id: str) -> Path:
        return self._current_dir(project_id) / "canvas.json"

    def _graph_path(self, project_id: str) -> Path:
        return self._current_dir(project_id) / "graph.json"

    def _runtime_path(self, project_id: str) -> Path:
        return self._current_dir(project_id) / "runtime.json"

    def _wal_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "wal" / "ops.jsonl"

    def _commits_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "commits"

    def _commit_dir(self, project_id: str, commit_id: str) -> Path:
        return self._commits_dir(project_id) / commit_id

    def _snapshots_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "snapshots"

    def _snapshot_dir(self, project_id: str, snapshot_id: str) -> Path:
        return self._snapshots_dir(project_id) / snapshot_id

    def _current_preview_meta_path(self, project_id: str) -> Path:
        return self._current_dir(project_id) / "preview.meta.json"

    def _current_preview_image_path(self, project_id: str, image_name: str) -> Path:
        return self._current_dir(project_id) / image_name

    def _commit_meta_path(self, project_id: str, commit_id: str) -> Path:
        return self._commit_dir(project_id, commit_id) / "meta.json"

    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def create_project(self, *, name: Optional[str] = None, project_id: Optional[str] = None) -> Dict[str, Any]:
        now = _utc_now_iso()
        if project_id:
            pid = str(project_id).strip()
        else:
            stem = _slugify_name(name or "")
            pid = f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{stem}-{_rand_suffix(4)}"
        pdir = self._project_dir(pid)
        if pdir.exists():
            raise FileExistsError(f"project already exists: {pid}")
        (pdir / "current").mkdir(parents=True, exist_ok=True)
        (pdir / "wal").mkdir(parents=True, exist_ok=True)
        (pdir / "commits").mkdir(parents=True, exist_ok=True)
        (pdir / "snapshots").mkdir(parents=True, exist_ok=True)
        meta = {
            "schemaVersion": self.SCHEMA_VERSION,
            "projectId": pid,
            "name": (name or pid).strip() if (name or "").strip() else pid,
            "createdAt": now,
            "updatedAt": now,
            "lastSavedAt": None,
            "currentRef": None,
            "commitCount": 0,
            "currentPreviewUpdatedAt": None,
            "stats": {
                "strokeCount": 0,
                "blockCount": 0,
                "fragmentCount": 0,
            },
        }
        _atomic_write_json(self._meta_path(pid), meta)
        _atomic_write_json(self._canvas_path(pid), {"schemaVersion": self.SCHEMA_VERSION})
        _atomic_write_json(self._graph_path(pid), {"schemaVersion": self.SCHEMA_VERSION})
        _atomic_write_json(self._runtime_path(pid), {"schemaVersion": self.SCHEMA_VERSION})
        self.append_wal(pid, "project_create", {"name": meta["name"]})
        return meta

    def list_projects(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            meta = self._read_json(child / "meta.json", {})
            if not isinstance(meta, dict):
                meta = {}
            current_canvas = self._read_json(child / "current" / "canvas.json", {})
            current_graph = self._read_json(child / "current" / "graph.json", {})
            current_preview_meta = self._read_json(child / "current" / "preview.meta.json", {})
            stats = dict(meta.get("stats") or {})
            if not stats:
                strokes = ((current_canvas or {}).get("strokes") or []) if isinstance(current_canvas, dict) else []
                graph_blocks = ((current_graph or {}).get("graphState") or {}).get("blocks") or {}
                graph_frags = ((current_graph or {}).get("graphState") or {}).get("fragments") or {}
                stats = {
                    "strokeCount": len(strokes) if isinstance(strokes, list) else 0,
                    "blockCount": len(graph_blocks) if isinstance(graph_blocks, dict) else 0,
                    "fragmentCount": len(graph_frags) if isinstance(graph_frags, dict) else 0,
                }
            items.append(
                {
                    "projectId": str(meta.get("projectId") or child.name),
                    "name": str(meta.get("name") or child.name),
                    "createdAt": meta.get("createdAt"),
                    "updatedAt": meta.get("updatedAt"),
                    "lastSavedAt": meta.get("lastSavedAt"),
                    "commitCount": int(meta.get("commitCount") or 0),
                    "currentPreviewUpdatedAt": meta.get("currentPreviewUpdatedAt"),
                    "stats": stats,
                    "currentPreview": current_preview_meta if isinstance(current_preview_meta, dict) and current_preview_meta else None,
                }
            )
        items.sort(key=lambda item: str(item.get("updatedAt") or ""), reverse=True)
        return items

    def load_project(self, project_id: str) -> Dict[str, Any]:
        pid = str(project_id).strip()
        pdir = self._project_dir(pid)
        if not pdir.exists():
            raise FileNotFoundError(f"project not found: {pid}")
        return {
            "meta": self._read_json(self._meta_path(pid), {}),
            "canvas": self._read_json(self._canvas_path(pid), {}),
            "graph": self._read_json(self._graph_path(pid), {}),
            "runtime": self._read_json(self._runtime_path(pid), {}),
            "current": self.get_current_summary(pid),
            "commits": self.list_commits(pid),
            "snapshots": self.list_snapshots(pid),
        }

    def _decode_snapshot_bytes(self, snapshot: Dict[str, Any]) -> Tuple[bytes, str, str]:
        data_b64 = str(snapshot.get("data") or "").strip()
        if not data_b64:
            raise ValueError("snapshot.data required")
        mime = str(snapshot.get("mime") or "image/jpeg")
        ext = ".png" if "png" in mime else ".jpg"
        try:
            image_bytes = base64.b64decode(data_b64)
        except Exception as exc:
            raise ValueError(f"invalid snapshot base64: {exc}") from exc
        return image_bytes, mime, ext

    def _snapshot_meta_dict(self, *, snapshot_id: str, snapshot: Dict[str, Any], mime: str, image_name: str, note: Optional[str] = None) -> Dict[str, Any]:
        return {
            "snapshotId": snapshot_id,
            "createdAt": _utc_now_iso(),
            "note": (note or "").strip()[:200] or None,
            "mime": mime,
            "width": int(snapshot.get("width") or 0),
            "height": int(snapshot.get("height") or 0),
            "bbox": snapshot.get("bbox"),
            "imageFile": image_name,
        }

    def _find_preview_image(self, dir_path: Path) -> Optional[Path]:
        for cand in ("preview.png", "preview.jpg", "preview.jpeg"):
            p = dir_path / cand
            if p.exists():
                return p
        return None

    def get_current_summary(self, project_id: str) -> Dict[str, Any]:
        pid = str(project_id).strip()
        meta = self._read_json(self._meta_path(pid), {})
        preview_meta = self._read_json(self._current_preview_meta_path(pid), {})
        if not isinstance(preview_meta, dict):
            preview_meta = {}
        return {
            "projectId": pid,
            "currentRef": (meta or {}).get("currentRef") if isinstance(meta, dict) else None,
            "preview": preview_meta or None,
            "previewUpdatedAt": (meta or {}).get("currentPreviewUpdatedAt") if isinstance(meta, dict) else None,
            "lastSavedAt": (meta or {}).get("lastSavedAt") if isinstance(meta, dict) else None,
        }

    def list_snapshots(self, project_id: str, *, limit: int = 60) -> List[Dict[str, Any]]:
        pid = str(project_id).strip()
        sdir = self._snapshots_dir(pid)
        if not sdir.exists():
            return []
        items: List[Dict[str, Any]] = []
        for child in sdir.iterdir():
            if not child.is_dir():
                continue
            meta = self._read_json(child / "meta.json", {})
            if not isinstance(meta, dict):
                continue
            image_file = None
            for cand in ("preview.png", "preview.jpg", "preview.jpeg"):
                p = child / cand
                if p.exists():
                    image_file = p.name
                    break
            items.append(
                {
                    "snapshotId": str(meta.get("snapshotId") or child.name),
                    "createdAt": meta.get("createdAt"),
                    "note": meta.get("note"),
                    "mime": meta.get("mime"),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "bbox": meta.get("bbox"),
                    "imageFile": image_file,
                }
            )
        items.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
        return items[: max(1, int(limit))]

    def save_snapshot(
        self,
        project_id: str,
        *,
        snapshot: Dict[str, Any],
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        pid = str(project_id).strip()
        pdir = self._project_dir(pid)
        if not pdir.exists():
            raise FileNotFoundError(f"project not found: {pid}")
        image_bytes, mime, ext = self._decode_snapshot_bytes(snapshot)
        snap_id = f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{_rand_suffix(4)}"
        sdir = self._snapshot_dir(pid, snap_id)
        sdir.mkdir(parents=True, exist_ok=True)
        image_name = f"preview{ext}"
        image_path = sdir / image_name
        _atomic_write_bytes(image_path, image_bytes)
        meta = self._snapshot_meta_dict(snapshot_id=snap_id, snapshot=snapshot, mime=mime, image_name=image_name, note=note)
        _atomic_write_json(sdir / "meta.json", meta)
        self.append_wal(pid, "project_snapshot_save", {"snapshotId": snap_id, "note": meta.get("note")})
        # bump project meta updated time
        self.save_current(pid, wal_op=None, meta_patch={})
        return meta

    def save_current_preview(
        self,
        project_id: str,
        *,
        snapshot: Dict[str, Any],
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        pid = str(project_id).strip()
        pdir = self._project_dir(pid)
        if not pdir.exists():
            raise FileNotFoundError(f"project not found: {pid}")
        image_bytes, mime, ext = self._decode_snapshot_bytes(snapshot)
        cur_dir = self._current_dir(pid)
        cur_dir.mkdir(parents=True, exist_ok=True)
        # Remove previous preview variant to keep a single canonical current preview.
        for old in ("preview.png", "preview.jpg", "preview.jpeg"):
            try:
                (cur_dir / old).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
        image_name = f"preview{ext}"
        _atomic_write_bytes(self._current_preview_image_path(pid, image_name), image_bytes)
        meta = {
            "kind": "currentPreview",
            "projectId": pid,
            "updatedAt": _utc_now_iso(),
            "note": (note or "").strip()[:200] or None,
            "mime": mime,
            "width": int(snapshot.get("width") or 0),
            "height": int(snapshot.get("height") or 0),
            "bbox": snapshot.get("bbox"),
            "imageFile": image_name,
        }
        _atomic_write_json(self._current_preview_meta_path(pid), meta)
        self.save_current(
            pid,
            wal_op=None,
            meta_patch={"currentPreviewUpdatedAt": meta["updatedAt"]},
            touch_last_saved=False,
        )
        self.append_wal(pid, "project_current_preview_save", {"note": meta.get("note")})
        return meta

    def read_current_preview_image(self, project_id: str) -> Tuple[bytes, str]:
        pid = str(project_id).strip()
        cdir = self._current_dir(pid)
        if not cdir.exists():
            raise FileNotFoundError(f"project not found: {pid}")
        meta = self._read_json(self._current_preview_meta_path(pid), {})
        image_path = None
        image_file = ""
        if isinstance(meta, dict):
            image_file = str(meta.get("imageFile") or "").strip()
            if image_file:
                p = cdir / image_file
                if p.exists():
                    image_path = p
        if image_path is None:
            image_path = self._find_preview_image(cdir)
        if image_path is None:
            raise FileNotFoundError(f"current preview not found: {pid}")
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        if isinstance(meta, dict) and str(meta.get("mime") or "").strip():
            mime = str(meta.get("mime"))
        return image_path.read_bytes(), mime

    def create_commit(
        self,
        project_id: str,
        *,
        canvas: Dict[str, Any],
        graph: Dict[str, Any],
        runtime: Dict[str, Any],
        snapshot: Dict[str, Any],
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        pid = str(project_id).strip()
        pdir = self._project_dir(pid)
        if not pdir.exists():
            raise FileNotFoundError(f"project not found: {pid}")
        commit_id = f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{_rand_suffix(6)}"
        cdir = self._commit_dir(pid, commit_id)
        cdir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(cdir / "canvas.json", canvas)
        _atomic_write_json(cdir / "graph.json", graph)
        _atomic_write_json(cdir / "runtime.json", runtime)
        image_bytes, mime, ext = self._decode_snapshot_bytes(snapshot)
        image_name = f"preview{ext}"
        _atomic_write_bytes(cdir / image_name, image_bytes)
        commit_meta = {
            "schemaVersion": self.SCHEMA_VERSION,
            "commitId": commit_id,
            "projectId": pid,
            "createdAt": _utc_now_iso(),
            "message": (message or "").strip()[:500] or None,
            "mime": mime,
            "width": int(snapshot.get("width") or 0),
            "height": int(snapshot.get("height") or 0),
            "bbox": snapshot.get("bbox"),
            "imageFile": image_name,
        }
        _atomic_write_json(cdir / "meta.json", commit_meta)
        # Commit updates currentRef + commitCount and timestamps.
        meta = self._read_json(self._meta_path(pid), {})
        if not isinstance(meta, dict):
            meta = {"projectId": pid, "schemaVersion": self.SCHEMA_VERSION}
        now = _utc_now_iso()
        meta["updatedAt"] = now
        meta["lastSavedAt"] = now
        meta["currentRef"] = commit_id
        try:
            count = int(meta.get("commitCount") or 0)
        except Exception:
            count = 0
        meta["commitCount"] = max(count + 1, len(self.list_commits(pid)))
        _atomic_write_json(self._meta_path(pid), meta)
        self.append_wal(pid, "project_commit_create", {"commitId": commit_id, "message": commit_meta.get("message")})
        return commit_meta

    def list_commits(self, project_id: str, *, limit: int = 200) -> List[Dict[str, Any]]:
        pid = str(project_id).strip()
        cdir = self._commits_dir(pid)
        if not cdir.exists():
            return []
        items: List[Dict[str, Any]] = []
        for child in cdir.iterdir():
            if not child.is_dir():
                continue
            meta = self._read_json(child / "meta.json", {})
            if not isinstance(meta, dict):
                continue
            image_file = str(meta.get("imageFile") or "").strip()
            if not image_file:
                img = self._find_preview_image(child)
                image_file = img.name if img else None  # type: ignore[assignment]
            items.append(
                {
                    "commitId": str(meta.get("commitId") or child.name),
                    "createdAt": meta.get("createdAt"),
                    "message": meta.get("message"),
                    "mime": meta.get("mime"),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "bbox": meta.get("bbox"),
                    "imageFile": image_file,
                }
            )
        items.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
        return items[: max(1, int(limit))]

    def load_commit(self, project_id: str, commit_id: str) -> Dict[str, Any]:
        pid = str(project_id).strip()
        cid = str(commit_id).strip()
        cdir = self._commit_dir(pid, cid)
        if not cdir.exists():
            raise FileNotFoundError(f"commit not found: {pid}/{cid}")
        return {
            "meta": self._read_json(cdir / "meta.json", {}),
            "canvas": self._read_json(cdir / "canvas.json", {}),
            "graph": self._read_json(cdir / "graph.json", {}),
            "runtime": self._read_json(cdir / "runtime.json", {}),
        }

    def read_commit_preview_image(self, project_id: str, commit_id: str) -> Tuple[bytes, str]:
        pid = str(project_id).strip()
        cid = str(commit_id).strip()
        cdir = self._commit_dir(pid, cid)
        if not cdir.exists():
            raise FileNotFoundError(f"commit not found: {pid}/{cid}")
        meta = self._read_json(cdir / "meta.json", {})
        image_path = None
        if isinstance(meta, dict):
            image_file = str(meta.get("imageFile") or "").strip()
            if image_file:
                p = cdir / image_file
                if p.exists():
                    image_path = p
        if image_path is None:
            image_path = self._find_preview_image(cdir)
        if image_path is None:
            raise FileNotFoundError(f"commit preview not found: {pid}/{cid}")
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        if isinstance(meta, dict) and str(meta.get("mime") or "").strip():
            mime = str(meta.get("mime"))
        return image_path.read_bytes(), mime

    def checkout_commit_to_current(self, project_id: str, commit_id: str) -> Dict[str, Any]:
        pid = str(project_id).strip()
        cid = str(commit_id).strip()
        bundle = self.load_commit(pid, cid)
        canvas = bundle.get("canvas") if isinstance(bundle.get("canvas"), dict) else {}
        graph = bundle.get("graph") if isinstance(bundle.get("graph"), dict) else {}
        runtime = bundle.get("runtime") if isinstance(bundle.get("runtime"), dict) else {}
        self.save_current(
            pid,
            canvas=canvas,
            graph=graph,
            runtime=runtime,
            wal_op="project_commit_checkout",
            wal_payload={"commitId": cid},
            meta_patch={"currentRef": cid},
        )
        try:
            data, mime = self.read_commit_preview_image(pid, cid)
            cmeta = bundle.get("meta") if isinstance(bundle.get("meta"), dict) else {}
            ext = ".png" if "png" in mime else ".jpg"
            cur_dir = self._current_dir(pid)
            cur_dir.mkdir(parents=True, exist_ok=True)
            for old in ("preview.png", "preview.jpg", "preview.jpeg"):
                try:
                    (cur_dir / old).unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
            image_name = f"preview{ext}"
            _atomic_write_bytes(cur_dir / image_name, data)
            preview_meta = {
                "kind": "currentPreview",
                "projectId": pid,
                "updatedAt": _utc_now_iso(),
                "note": f"checkout:{cid}",
                "mime": mime,
                "width": int(cmeta.get("width") or 0) if isinstance(cmeta, dict) else 0,
                "height": int(cmeta.get("height") or 0) if isinstance(cmeta, dict) else 0,
                "bbox": cmeta.get("bbox") if isinstance(cmeta, dict) else None,
                "imageFile": image_name,
            }
            _atomic_write_json(self._current_preview_meta_path(pid), preview_meta)
            self.save_current(pid, wal_op=None, meta_patch={"currentPreviewUpdatedAt": preview_meta["updatedAt"]}, touch_last_saved=False)
        except Exception:
            pass
        return bundle

    def delete_commit(self, project_id: str, commit_id: str) -> Dict[str, Any]:
        pid = str(project_id).strip()
        cid = str(commit_id).strip()
        cdir = self._commit_dir(pid, cid)
        if not cdir.exists():
            raise FileNotFoundError(f"commit not found: {pid}/{cid}")
        shutil.rmtree(cdir)
        meta = self._read_json(self._meta_path(pid), {})
        current_ref_cleared = False
        if not isinstance(meta, dict):
            meta = {"projectId": pid, "schemaVersion": self.SCHEMA_VERSION}
        if str(meta.get("currentRef") or "") == cid:
            meta["currentRef"] = None
            current_ref_cleared = True
        meta["commitCount"] = len(self.list_commits(pid))
        meta["updatedAt"] = _utc_now_iso()
        _atomic_write_json(self._meta_path(pid), meta)
        self.append_wal(pid, "project_commit_delete", {"commitId": cid})
        return {"currentRefCleared": current_ref_cleared, "commitCount": meta.get("commitCount")}

    def delete_project(self, project_id: str) -> None:
        pid = str(project_id).strip()
        pdir = self._project_dir(pid)
        if not pdir.exists():
            raise FileNotFoundError(f"project not found: {pid}")
        shutil.rmtree(pdir)

    def read_snapshot_image(self, project_id: str, snapshot_id: str) -> Tuple[bytes, str]:
        pid = str(project_id).strip()
        sid = str(snapshot_id).strip()
        sdir = self._snapshot_dir(pid, sid)
        if not sdir.exists():
            raise FileNotFoundError(f"snapshot not found: {pid}/{sid}")
        meta = self._read_json(sdir / "meta.json", {})
        if not isinstance(meta, dict):
            meta = {}
        image_file = str(meta.get("imageFile") or "").strip()
        if not image_file:
            for cand in ("preview.png", "preview.jpg", "preview.jpeg"):
                if (sdir / cand).exists():
                    image_file = cand
                    break
        if not image_file:
            raise FileNotFoundError(f"snapshot image not found: {pid}/{sid}")
        path = sdir / image_file
        if not path.exists():
            raise FileNotFoundError(f"snapshot image file missing: {pid}/{sid}/{image_file}")
        mime = str(meta.get("mime") or ("image/png" if path.suffix.lower() == ".png" else "image/jpeg"))
        return path.read_bytes(), mime

    def append_wal(self, project_id: str, op_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        pid = str(project_id).strip()
        row = {
            "ts": _utc_now_iso(),
            "op": str(op_type or "").strip() or "unknown",
            "payload": payload or {},
        }
        _append_jsonl(self._wal_path(pid), row)

    def save_current(
        self,
        project_id: str,
        *,
        canvas: Optional[Dict[str, Any]] = None,
        graph: Optional[Dict[str, Any]] = None,
        runtime: Optional[Dict[str, Any]] = None,
        wal_op: Optional[str] = None,
        wal_payload: Optional[Dict[str, Any]] = None,
        meta_patch: Optional[Dict[str, Any]] = None,
        touch_last_saved: bool = True,
    ) -> Dict[str, Any]:
        pid = str(project_id).strip()
        pdir = self._project_dir(pid)
        if not pdir.exists():
            raise FileNotFoundError(f"project not found: {pid}")
        if canvas is not None:
            _atomic_write_json(self._canvas_path(pid), canvas)
        if graph is not None:
            _atomic_write_json(self._graph_path(pid), graph)
        if runtime is not None:
            _atomic_write_json(self._runtime_path(pid), runtime)
        if wal_op:
            self.append_wal(pid, wal_op, wal_payload or {})

        meta = self._read_json(self._meta_path(pid), {})
        if not isinstance(meta, dict):
            meta = {}
        now = _utc_now_iso()
        meta.setdefault("schemaVersion", self.SCHEMA_VERSION)
        meta.setdefault("projectId", pid)
        meta.setdefault("name", pid)
        meta.setdefault("currentRef", None)
        meta.setdefault("commitCount", 0)
        meta.setdefault("currentPreviewUpdatedAt", None)
        meta["updatedAt"] = now
        if touch_last_saved:
            meta["lastSavedAt"] = now
        if isinstance(meta_patch, dict):
            for k, v in meta_patch.items():
                if k == "stats" and isinstance(v, dict):
                    meta["stats"] = {**(meta.get("stats") or {}), **v}
                else:
                    meta[k] = v
        _atomic_write_json(self._meta_path(pid), meta)
        return meta
