from __future__ import annotations

import base64
import json
import os
from typing import Dict, List, Optional

from semantic_graph import VisionBackend, VisionPayload, VisionResult

from app.llm_client import call_chat_completions


DEFAULT_LLM_MODEL = os.getenv("GRAPH_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"


def normalize_vision_image_mode(value: Optional[str]) -> str:
    token = str(value or "").strip().lower()
    if token in {"off", "auto", "always"}:
        return token
    return "auto"


class NoopVisionBackend(VisionBackend):
    """Placeholder vision backend."""

    def analyze(self, payload: VisionPayload) -> List[VisionResult]:
        return []


VISION_SYSTEM_PROMPT = (
    "You are a diagram-understanding assistant for a collaborative canvas. "
    "Given a group of stroke fragments plus the nearby blocks, decide whether the strokes "
    "should be merged into an existing block or promoted as a brand new diagram block. "
    "Always return JSON: {\"results\": [{\"decision\": \"merge_block\"|\"new_block\", "
    "\"target_block_id\": str?, \"label\": str?, \"summary\": str?, \"confidence\": float(0..1), "
    "\"relationships\": [{\"type\": str, \"target\": str, \"score\": float?}]?}]}. "
    "Only reference block IDs provided in the context. Always include confidence for each result. "
    "If unsure, prefer \"new_block\" with a cautious summary and lower confidence."
)

VISION_MANUAL_PROMOTE_PROMPT = (
    "You are a canvas interpreter for a collaborative canvas. "
    "Assume it should become a new block and focus on understanding the drawing's likely meaning/purpose. "
    "For example, if it looks like a cluster of shapes and lines around a few text blocks, it might be a diagram illustrating the relationship between those text blocks. "
    "Another example is a cat, car, or person drawing. "
    "Prioritize semantic interpretation over geometric narration. Avoid generic summaries like "
    "\"a collection of lines and curves\" unless the meaning is truly unknowable. "
    "Always return JSON: {\"results\": [{\"decision\": \"new_block\", "
    "\"target_block_id\": null, \"label\": str, \"summary\": str, \"confidence\": float(0..1), "
    "\"relationships\": [{\"type\": str, \"target\": str, \"score\": float?}]?}]}. "
    "When an image is attached, use it as the primary evidence; detailed stroke lists may be omitted. "
    "Use a concise, useful label (prefer < 40 chars). Summary should capture what the diagram means or what role it plays. "
    "Only reference block IDs provided in candidateBlocks. If uncertain, state the best plausible interpretation and lower confidence."
)


class LLMVisionBackend(VisionBackend):
    def __init__(
        self,
        model: Optional[str] = None,
        *,
        max_tokens: int = 1200,
        image_mode: str = "auto",
    ) -> None:
        self.model = model or os.getenv("VISION_MODEL") or DEFAULT_LLM_MODEL
        self.max_tokens = max_tokens
        self.image_mode = normalize_vision_image_mode(image_mode)

    def set_image_mode(self, mode: Optional[str]) -> None:
        self.image_mode = normalize_vision_image_mode(mode)

    def analyze(self, payload: VisionPayload) -> List[VisionResult]:
        include_image, image_reason = self._should_attach_image(payload)
        context = self._build_context(payload, include_image=include_image, image_reason=image_reason)
        system_prompt = self._select_system_prompt(payload)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        try:
            parsed, _ = call_chat_completions(messages, model=self.model, max_tokens=self.max_tokens)
        except Exception:
            return []
        results_raw = []
        if isinstance(parsed, dict):
            if isinstance(parsed.get("results"), list):
                results_raw = parsed["results"]
            else:
                results_raw = parsed.get("items") or []
        elif isinstance(parsed, list):
            results_raw = parsed
        results: List[VisionResult] = []
        for item in results_raw or []:
            try:
                results.append(
                    VisionResult(
                        kind=str(item.get("kind") or ""),
                        decision=item.get("decision") or item.get("kind"),
                        label=item.get("label"),
                        stroke_fragment_ids=list(item.get("stroke_fragment_ids") or payload.stroke_fragment_ids),
                        target_fragment_id=item.get("target_fragment_id"),
                        target_block_id=item.get("target_block_id"),
                        confidence=_safe_float(item.get("confidence")),
                        summary=item.get("summary"),
                        relationships=item.get("relationships"),
                        extra={
                            k: v
                            for k, v in item.items()
                            if k
                            not in {
                                "kind",
                                "decision",
                                "label",
                                "stroke_fragment_ids",
                                "target_fragment_id",
                                "target_block_id",
                                "confidence",
                                "summary",
                                "relationships",
                            }
                        },
                    )
                )
            except Exception:
                continue
        return results

    def _build_context(
        self,
        payload: VisionPayload,
        *,
        include_image: bool = False,
        image_reason: str = "disabled",
    ) -> Dict[str, object]:
        group_reason = self._group_reason(payload)
        group_meta: Dict[str, object] = {}
        snapshot_meta: Optional[Dict[str, object]] = None
        for item in payload.metadata or []:
            if not group_meta and isinstance(item, dict) and "group_id" in item:
                group_meta = item
            maybe_snapshot = item.get("snapshot") if isinstance(item, dict) else None
            if isinstance(maybe_snapshot, dict):
                snapshot_meta = maybe_snapshot
        manual_image_only = include_image and group_reason == "manual_promote"
        strokes_payload = (
            []
            if manual_image_only
            else (self._compact_fragments(payload.fragments) if include_image else payload.fragments)
        )
        candidate_blocks = self._compact_candidate_blocks(payload.candidate_blocks)
        context: Dict[str, object] = {
            "visionImageMode": self.image_mode,
            "visionImageAttached": include_image,
            "visionImageReason": image_reason,
            "visionTask": "manual_promote_semantic" if group_reason == "manual_promote" else "routing",
        }
        if snapshot_meta:
            context["snapshot"] = snapshot_meta
        context.update(
            {
                "group": {
                    "id": group_meta.get("group_id"),
                    "reason": group_meta.get("reason"),
                    "bbox": group_meta.get("bbox"),
                    "count": group_meta.get("count"),
                },
                "strokes": strokes_payload,
                "candidateBlocks": candidate_blocks,
            }
        )
        if manual_image_only:
            context["strokesOmitted"] = True
            context["strokeSummary"] = self._summarize_fragments(payload.fragments)
        if include_image and payload.image_bytes:
            try:
                context["_image_data"] = base64.b64encode(payload.image_bytes).decode("ascii")
                context["_image_mime"] = payload.image_mime or "image/png"
            except Exception:
                context["visionImageAttached"] = False
                context["visionImageReason"] = "encode_failed"
        return {**context}

    def _should_attach_image(self, payload: VisionPayload) -> tuple[bool, str]:
        mode = normalize_vision_image_mode(self.image_mode)
        if mode == "off":
            return False, "mode_off"
        if not payload.image_bytes:
            return False, "no_snapshot"
        if mode == "always":
            return True, "mode_always"
        if self._group_reason(payload) == "manual_promote":
            return True, "auto_manual_promote"

        point_total = 0
        tool_kinds: set[str] = set()
        pen_like = 0
        for frag in payload.fragments or []:
            tool = str(frag.get("tool") or "").lower()
            if tool:
                tool_kinds.add(tool)
            pts = frag.get("points")
            if isinstance(pts, list):
                point_total += len(pts)
                if tool in {"pen", "polyline"}:
                    pen_like += 1
        overlaps: List[float] = []
        for block in payload.candidate_blocks or []:
            try:
                overlaps.append(float(block.get("overlap") or 0.0))
            except Exception:
                overlaps.append(0.0)
        overlaps.sort(reverse=True)
        ambiguous_candidates = (
            len(overlaps) >= 2 and overlaps[0] > 0.0 and abs(overlaps[0] - overlaps[1]) < 0.12
        )
        complex_shape = (
            point_total >= 100
            or (pen_like >= 2 and point_total >= 70)
            or (len(tool_kinds) >= 3 and point_total >= 60)
        )
        sparse_context_complex = len(payload.candidate_blocks or []) == 0 and point_total >= 70
        if ambiguous_candidates:
            return True, "auto_ambiguous_candidates"
        if complex_shape:
            return True, "auto_complex_shape"
        if sparse_context_complex:
            return True, "auto_sparse_context_complex"
        return False, "auto_simple"

    def _group_reason(self, payload: VisionPayload) -> str:
        for item in payload.metadata or []:
            if isinstance(item, dict) and item.get("group_id") is not None:
                return str(item.get("reason") or "").strip().lower()
        return ""

    def _select_system_prompt(self, payload: VisionPayload) -> str:
        if self._group_reason(payload) == "manual_promote":
            return VISION_MANUAL_PROMOTE_PROMPT
        return VISION_SYSTEM_PROMPT

    def _compact_candidate_blocks(self, candidate_blocks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        compacted: List[Dict[str, object]] = []
        for block in list(candidate_blocks or [])[:3]:
            summary = str(block.get("summary") or "").strip()
            if len(summary) > 120:
                summary = f"{summary[:120]}..."
            compacted.append(
                {
                    "blockId": block.get("blockId"),
                    "label": block.get("label"),
                    "summary": summary,
                    "bbox": block.get("bbox"),
                    "overlap": block.get("overlap"),
                }
            )
        return compacted

    def _compact_fragments(self, fragments: List[Dict[str, object]]) -> List[Dict[str, object]]:
        compacted: List[Dict[str, object]] = []
        for frag in fragments or []:
            item: Dict[str, object] = {
                "id": frag.get("id"),
                "tool": frag.get("tool"),
                "bbox": frag.get("bbox"),
            }
            points = frag.get("points")
            if isinstance(points, list):
                item["pointCount"] = len(points)
                tool = str(frag.get("tool") or "").lower()
                if tool in {"line"} and len(points) >= 2:
                    item["endpoints"] = [self._xy(points[0]), self._xy(points[-1])]
                elif tool in {"pen", "polyline"}:
                    item["sampledPoints"] = self._sample_points(points, limit=8)
            style = frag.get("style")
            if isinstance(style, dict):
                lite_style = {}
                if style.get("size") is not None:
                    lite_style["size"] = style.get("size")
                if style.get("color") is not None:
                    lite_style["color"] = style.get("color")
                if lite_style:
                    item["style"] = lite_style
            compacted.append(item)
        return compacted

    def _summarize_fragments(self, fragments: List[Dict[str, object]]) -> Dict[str, object]:
        tool_counts: Dict[str, int] = {}
        point_total = 0
        bbox_union: Optional[List[float]] = None
        pen_count = 0
        line_count = 0
        closed_like = 0
        for frag in fragments or []:
            tool = str(frag.get("tool") or "").lower() or "unknown"
            tool_counts[tool] = int(tool_counts.get(tool, 0)) + 1
            if tool in {"pen", "polyline"}:
                pen_count += 1
            if tool == "line":
                line_count += 1
            points = frag.get("points")
            if isinstance(points, list):
                point_total += len(points)
                if len(points) >= 3:
                    try:
                        p0 = self._xy(points[0])
                        pn = self._xy(points[-1])
                        if abs(p0[0] - pn[0]) <= 2 and abs(p0[1] - pn[1]) <= 2:
                            closed_like += 1
                    except Exception:
                        pass
            bbox = frag.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
                except Exception:
                    continue
                if bbox_union is None:
                    bbox_union = [x0, y0, x1, y1]
                else:
                    bbox_union[0] = min(bbox_union[0], x0)
                    bbox_union[1] = min(bbox_union[1], y0)
                    bbox_union[2] = max(bbox_union[2], x1)
                    bbox_union[3] = max(bbox_union[3], y1)
        summary: Dict[str, object] = {
            "count": len(fragments or []),
            "toolCounts": tool_counts,
            "pointTotal": point_total,
            "penLikeCount": pen_count,
            "lineCount": line_count,
        }
        if closed_like:
            summary["closedLikeCount"] = closed_like
        if bbox_union is not None:
            summary["bbox"] = [round(v, 1) for v in bbox_union]
        return summary

    def _sample_points(self, points: List[object], *, limit: int = 8) -> List[List[int]]:
        if not isinstance(points, list) or not points:
            return []
        if len(points) <= limit:
            return [self._xy(p) for p in points]
        out: List[List[int]] = []
        used_idx = set()
        last_index = len(points) - 1
        for i in range(limit):
            idx = round((last_index * i) / max(1, limit - 1))
            if idx in used_idx:
                continue
            used_idx.add(idx)
            out.append(self._xy(points[idx]))
        return out

    def _xy(self, point: object) -> List[int]:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                return [int(round(float(point[0]))), int(round(float(point[1])))]
            except Exception:
                return [0, 0]
        return [0, 0]


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
