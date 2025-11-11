# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Literal, Optional, Tuple, Dict,Union, Any
from pydantic import BaseModel, Field

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, Optional[float]]
Point4 = Tuple[float, float, Optional[float], Optional[float]]
# 与前端一致的颜色枚举
ColorName = Literal[
    "black","blue","green","grey",
    "light-blue","light-green","light-red","light-violet",
    "orange","red","violet","white","yellow"
]

class StrokeStyle(BaseModel):
    # 画笔样式，默认给出可复现的值
    size: Optional[Literal["s","m","l","xl"]] = "m"
    color: Optional[ColorName] = "black"
    opacity: Optional[float] = 1.0

class AIStrokeV11(BaseModel):
    """
    tool:
      - 'pen','line','poly','ellipse',... 以及现在新增的 'text'
    'text' 约定：
      - points: [[x,y],[x+w,y+h]]  // 左上角→右下角
      - style.color 作为字体颜色；style.size/opacity保留协议即可
      - meta: {
          "text": str,
          "summary": str,
          "fontFamily": str,
          "fontWeight": str,   # '400'/'bold'/...
          "fontSize": float,   # px
          "growDir": str       # 'down'|'right'|'up'|'left'
        }
    """
    id: str
    tool: str  # 'pen' | 'line' | 'rect' | 'ellipse' | ...
    # points: [x, y, t?, pressure?]
    points: List[Union[Point2, Point3, Point4]]
    style: Optional[StrokeStyle] = None
    meta: Optional[Dict[str, Any]] = None

class CanvasInfo(BaseModel):
    # 画布 / 视口信息（可选）
    width: Optional[int] = None
    height: Optional[int] = None
    viewport: Optional[Tuple[float, float, float, float]] = None  # [x, y, w, h]

class AIStrokePayload(BaseModel):
    # 模型输入/输出使用的统一结构
    version: int = 1
    intent: Optional[Literal["complete","hint","alt"]] = "complete"
    replace: Optional[List[str]] = None
    canvas: Optional[CanvasInfo] = None
    strokes: List[AIStrokeV11] = Field(default_factory=list)

# —— 会话初始化 / 增量传输 —— #
class InitSessionRequest(BaseModel):
    mode: Literal["light_helper","work_assistant","Vision"] = "light_helper"
    init_goal: Optional[str] = None
    tags: Optional[List[str]] = None

class InitSessionResponse(BaseModel):
    sid: str
    note: Optional[str] = None

class DeltaPayload(BaseModel):
    strokes: List[AIStrokeV11] = Field(default_factory=list)


# ===== 接口请求/响应 =====

class SuggestRequest(BaseModel):
    # 兼容两种模式：
    # A) 旧格式：context（全量上下文）
    # B) 新格式：sid + delta（只传增量）
    mode: Optional[Literal["full", "light", "vision"]] = None
    sid: Optional[str] = None
    context: Optional[AIStrokePayload] = Field(None, description="旧格式：全量上下文")
    delta: Optional[DeltaPayload] = Field(None, description="新格式：增量笔画（只传最近新增）")
    # 人类指令/偏好，可选
    hint: Optional[str] = None
    # 临时覆盖模型与采样参数，可选
    model: Optional[str] = None
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 1024
    # 建议的 AI 单笔点数上限（用于引导模型控制规模）
    gen_scale: Optional[int] = Field(default=None, description="preferred max points for AI stroke")
    # 可选的Vision模式图像输入
    image_data: Optional[str] = None          # base64 无前缀，或完整 dataURL 也行（见下）
    image_mime: Optional[str] = "image/png"   # "image/png" | "image/jpeg"
    snapshot_size: Optional[tuple[int, int]] = None  # (w,h) 供日志/调试用（可选）
    # —— 新增：Vision 版本与序列号（2.0 二段式）
    vision_version: Optional[float] = 1.0
    seq: Optional[int] = 1  # 1 = Step-1（图像理解），2 = Step-2（注入full）
    # Step-2 时把 Step-1 的文字说明作为“强化指令”送入 full 流程
    instruction_text: Optional[str] = None

class SuggestResponse(BaseModel):
    ok: bool = True
    payload: AIStrokePayload
    usage: Optional[Dict[str, Any]] = None
    note: Optional[str] = None
    # Vision 2.0 额外携带结构化的分析/建议，供前端触发第二步
    vision2: Optional[Dict[str, Any]] = None

class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model: Optional[str] = None
    base_url: Optional[str] = None

# ===== 会话全量同步（把前端当前笔迹快照覆盖到会话） =====
class SyncSessionRequest(BaseModel):
    sid: str
    strokes: List[AIStrokeV11] = Field(default_factory=list)
    graph_snapshot: Optional["GraphSnapshotPayload"] = None

class SyncSessionResponse(BaseModel):
    ok: bool = True
    count: int = 0

class CompletionRequest(BaseModel):
    text: str
    max_tokens: Optional[int] = 4096

class CompletionResponse(BaseModel):
    completion: str


class GraphAutoModeRequest(BaseModel):
    sid: str
    enabled: bool
    canvas_size: Optional[Tuple[float, float]] = None
    strokes: Optional[List[AIStrokeV11]] = None
    graph_snapshot: Optional["GraphSnapshotPayload"] = None


class GraphAutoModeResponse(BaseModel):
    ok: bool = True
    enabled: bool = False


class GraphSnapshotPayload(BaseModel):
    bbox: Tuple[float, float, float, float]
    width: int
    height: int
    mime: Literal["image/png", "image/jpeg"]
    data: str


class GraphSnapshotResponse(BaseModel):
    blocks: List[Dict[str, Any]] = Field(default_factory=list)
    fragments: List[Dict[str, Any]] = Field(default_factory=list)
    groups: List[Dict[str, Any]] = Field(default_factory=list)


class PromoteGroupRequest(BaseModel):
    sid: str
    group_id: str


class PromoteGroupResponse(BaseModel):
    ok: bool = True
    block: Optional[Dict[str, Any]] = None

