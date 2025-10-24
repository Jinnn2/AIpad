# 队友看这里！
## 更新日志：
> 2025/10/24 增加了基本的text功能：输入框，允许llm返回一个输入框类型，允许用户自定义，拉取并创建一个输入框类型。
> 
> 待办：输入框大小自适应
> 
> 待办：输入框的选取，拖动与修改
> 
> 待办：llm的输入框调整与补全
> 
> 待办：构建llm指令集

# 🧠 AIpad — Intelligent Line & Note System

> “从线条到文字，从理解到生成”
> 
> A unified notebook that sees, understands, and completes both **drawings** and **notes**.



## 🚀 项目简介

**AIpad** 是一个基于 FastAPI + React 的多模态笔记/绘图系统。
它结合了 AI 对线稿的**视觉理解**与**结构化生成能力**，支持实时补笔、形状识别、块级管理与上下文推理。
当前版本实现了智能线稿画板（LineArtBoard），并正在拓展到文字-图画混合的全能笔记模式（AI Note）。



## 🧩 核心特性

| 模块                   | 功能说明                                                |
| -------------------- | --------------------------------------------------- |
| 🎨 **LineArtBoard**  | 无限画布，支持笔刷、橡皮、椭圆、撤销/重做等基础绘图操作                        |
| 🤖 **AI 补全引擎**       | 模型根据画布上下文预测下一笔，可在 `light` / `full` / `vision` 模式间切换 |
| 🪄 **Vision 2.0 模式** | 分两阶段执行：① 图像理解（analysis + instruction）② 结构化绘制        |
| 📡 **LLM Gateway**   | FastAPI 后端统一封装模型调用（支持 OpenAI / ChatAnywhere 协议）     |
| 💾 **Session Store** | 本地内存会话管理，支持增量同步、自动样例注入与上下文压缩                        |
| 🧱 **Prompt 构建系统**   | 根据模式自动生成提示词（light/full/vision），并对接多阶段 Pipeline      |
| 🪶 **日志体系**          | 全自动输入/输出记录，包含模型原文、清洗后 payload、图像快照等                 |

---

## 🧬 AI Note Pipeline 规划（开发中）

AIpad 的下一阶段目标是实现 **文字/图画混合型智能笔记系统** ——
一个能“识图、能补全、能理解上下文”的 **全能 Notebook**。

### 🔹 模块设计

| 模块            | 说明                                                        |
| ------------- | --------------------------------------------------------- |
| 🖼️ **图理解模块** | 对当前聚焦块进行快照识别，理解形状、布局与语义结构，输出块形状与任务指示。                     |
| 🧩 **块维护模块**  | 结合聚焦块的文字与笔画数据，更新标签（如类型、语义、层级）。                            |
| 🗂️ **归类模块**  | 维护后端块表，按语境组织上下文，形成持续演化的背景 Prompt。                         |
| 🚪 **入口模块**   | 判断下一步动作：<br>① 若需视觉理解，确定聚焦块；<br>② 若需生成，组织完整 Prompt 并打开目标块。 |
| 🔮 **预测模块**   | 调用模型生成标签化结果（图/文/混合），并由归类模块进行结构化整合。                        |


## 🧠 模式概览

| 模式                       | 说明                                    |
| ------------------------ | ------------------------------------- |
| `light_helper`           | 轻量预测单笔线条，快速、低延迟。                      |
| `full`                   | 上下文感知补全，生成多笔复杂线条。                     |
| `vision`                 | 多模态视觉理解，结合画布截图推理与生成。                  |
| `vision 2.0`             | 新一代两阶段视觉流程：**Step 1 分析 → Step 2 绘制**。 |
| `work_assistant` *(规划中)* | 用于综合理解画面与文字内容，支持多层笔记结构与思维导图式推理。       |


## 🏗️ 技术栈

**前端（/src）**

* ⚛️ React + TypeScript
* 🎨 Konva.js / Canvas 渲染
* 🧩 自定义 UI（SidePanel / TopToolbar / BottomPanel）
* 📡 Axios / Fetch 调用 FastAPI 接口

**后端（/app）**

* 🐍 FastAPI
* 🧠 OpenAI SDK / Chat Completions
* 🗃️ Pydantic Schema 校验（`AIStrokePayload v1.1`）
* 📜 Session 管理与 Prompt 生成系统
* 🔍 全自动日志（输入请求、模型原文、输出清洗后 JSON）


## 📂 主要目录结构

```
AIpad/
├── app/
│   ├── main.py             # FastAPI 主入口（/suggest, /session）
│   ├── schemas.py          # 数据结构定义（AIStrokePayload, SuggestRequest 等）
│   ├── prompting.py        # 模型提示词构建器（light/full/vision 模式）
│   ├── llm_client.py       # 调用与日志封装（OpenAI Chat API）
│   ├── session_store.py    # 内存会话管理、重采样与量化
│   └── ...
├── src/
│   ├── LineArtBoard.tsx    # 主画板逻辑
│   ├── LineArtUI.tsx       # 工具栏与操作面板组件
│   ├── App.tsx / main.tsx  # 前端入口
│   └── index.css / App.css # 全局样式
├── .env                    # 环境配置（API Key、代理、CORS等）
└── README.md
```

---

## 🧰 开发与运行

### 1️⃣ 启动后端

```bash
cd app
uvicorn app.main:app --reload --port 8000
```

### 2️⃣ 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173` 即可。

---

## 🔭 未来计划

* [ ] 新增 **输入框类** 与 **字体类**，实现图文混排与文字识别。
* [ ] 增强块级管理（块语义、层级、标签、上下文）。
* [ ] 统一图文 Pipeline，实现 AI Note 笔记理解与重构。
* [ ] 引入矢量化笔迹存储与历史可回放。
* [ ] 多 Agent 协同：视觉分析 Agent + 结构生成 Agent。

---

## 💡 项目理念

AIpad 的核心思想是：

> “让 AI 看懂你写的、画的、想的。”

它不仅是一个绘图助手，更是一个可以在**视觉空间中思考的智能笔记平台**。
从曲线到文字，从几何到语义——AIpad 正在尝试让「AI 理解」真正回到创作的原点。
