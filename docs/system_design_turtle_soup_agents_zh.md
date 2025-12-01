# Echoes of Deceit – 海龟汤多 Agent 系统设计

> 草稿日期：2025‑11‑30

## 1. 愿景与目标

本项目旨在实现一个**多 Agent “海龟汤”（情境猜谜）游戏系统**，其中 Agent 可以扮演不同的角色：

- **DM（主持人 / 旁白 / 裁判）** – 呈现谜题，控制信息流，管理世界状态，并负责根据规则对玩家的问题和假设进行裁决。
- **Player Agent（玩家 Agent）** – 提出 是/否/是也不是/无关 类型的问题，对谜底进行假设，并尝试收敛到真相。

该系统利用了：

- **RAG（检索增强生成）** 用于谜题知识和世界一致性（已在 `src/rag/` 下实现）。
- **长期记忆（Long‑term memory）** 用于跟踪游戏进度、玩家行为和跨会话的持久知识。
- **LangChain 和 LangGraph** 用于 Agent 编排和有状态的对话工作流。
- **可插拔的模型提供商** 支持 **Ollama（本地）** 和 **远程 API 模型**（例如 OpenAI 兼容端点）。

主要目标：

1. **模块化架构**，清晰分离谜题数据、游戏逻辑和 Agent 角色。
2. 支持**多个游戏 / 谜题**，具有隔离的知识库和记忆。
3. **可复现的游戏会话**（保存 / 加载状态）和可观测性（日志、追踪、回放）。
4. 易于扩展新角色、新 RAG 提供商和新模型后端。

---

## 2. 高层架构

从高层来看，系统由以下几层组成：

1. **接口层（Interface Layer）**
   - 渠道：CLI、Web UI、聊天（例如未来的 Discord/Slack 机器人）。
   - 负责用户认证（如果有）、会话管理，并将用户消息映射到 LangGraph 入口点。

2. **编排层（Orchestration Layer - LangGraph / LangChain）**
   - 定义代表游戏流程的 **Agent 图**：DM、Player Agent 及其交互。
   - 管理对话状态、路由和工具调用。

3. **游戏逻辑层（Game Logic Layer）**
   - 游戏会话管理器、谜题加载器、规则引擎、计分系统。
   - 与 RAG 层集成以回答内容查询。
   - 与记忆层集成以持久化和查询玩家/游戏历史。

4. **知识与记忆层（Knowledge and Memory Layer）**
   - **RAG 子系统**（已在 `src/rag/` 中实现）：
     - `KnowledgeBase`, `RAGProviderFactory`, `LightRAGProvider`, `MiniRAGProvider`, `GameDataLoader`。
   - **长期记忆**：
     - 每会话和每玩家的记忆存储。
     - 嵌入 + 向量存储（可以复用现有的 RAG 存储或单独的记忆存储）。

5. **模型提供商层（Model Provider Layer）**
   - LLM 和嵌入模型的抽象，可由以下支持：
     - Ollama
     - OpenAI 兼容的 HTTP API（OpenAI, Azure OpenAI, 其他提供商）。

6. **持久化与存储层（Persistence and Storage Layer）**
   - 基于文件的谜题、知识库和 RAG 制品存储（已存在于 `data/` 和 `rag_storage/` 下）。
   - 用于会话日志和长期记忆的数据库或文件。

7. **配置与可观测性（Configuration and Observability）**
   - 用于选择 RAG 提供商、LLM 后端和谜题集的中心配置文件（例如 YAML/JSON）。
   - 日志记录、指标钩子和 LangGraph 流程的可选追踪。

### 2.1 组件图（概念性）

主要组件和关系：

- `GameEngine`
   - 使用 `PuzzleRepository` 加载谜题元数据。
   - 与 `KnowledgeBaseManager`（现有 `KnowledgeBase` 的包装器）交互。
   - 生成 `GameSession` 实例。

- `GameSession`
   - 持有对活动 `PuzzleContext`、`MemoryManager` 和 `AgentGraph`（LangGraph 图）的引用。
   - 暴露 `handle_player_message`、`save`、`load` 等方法。

- `AgentGraph` (LangGraph)
   - 节点：`DMNode`, `PlayerNode`, `ToolInvocationNode`, `RAGQueryNode`, `MemoryUpdateNode`。
   - 边定义了取决于游戏状态的节点间流程。

- `MemoryManager`
   - 接口对接 **长期记忆存储**：
     - 情节记忆：特定于游戏。
     - 语义记忆：玩家画像、重复模式。

- `ModelProviderRegistry`
   - 基于配置（`provider=ollama` 或 `provider=api`）提供对 `LLMClient` 和 `EmbeddingClient` 的统一访问。

---

## 3. 游戏模型与领域概念

### 3.1 核心领域实体

- **Puzzle（谜题）**
   - `id`：唯一标识符（例如 `puzzle1`, `puzzle_1_en`）。
   - `title`：谜题标题。
   - `description`：高层描述。
   - `puzzle_statement`：展示给玩家的文本（汤面）。
   - `answer`：标准解答文本（汤底）。
   - `constraints`：特殊规则（例如最大提问数、问题类型）。
   - `tags`：难度、主题等。

- **Game（游戏）**
   - 可以是一对一映射到一个谜题（简单模式），也可以包含战役中的多个谜题（高级模式）。

- **GameSession（游戏会话）**
   - `session_id`：唯一会话标识符。
   - `puzzle_id`：关联的谜题。
   - `players`：一个或多个 `PlayerProfile` 对象。
   - `state`：`LOBBY`（大厅）, `IN_PROGRESS`（进行中）, `COMPLETED`（已完成）, `ABORTED`（已中止）。
   - `turn_history`：按时间顺序排列的回合列表，包括问题、答案和元事件。
   - `config`：此会话的 LLM / RAG / 记忆配置。

- **PlayerProfile（玩家画像）**
   - `player_id`（或 `user_id`）。
   - `display_name`。
   - `preferences`（难度、语言等）。
   - `long_term_memory_ref`：指向其记忆存储的 ID/指针。

- **AgentRole（Agent 角色）**
   - `DM`, `Player`, `Observer`（未来）, `HintMaster`（未来，专门的提示机器人）。

### 3.2 数据源

- **谜题 JSON 文件** 位于 `data/situation_puzzles/...`（例如 `puzzle_1_en.json`）：
   - 已使用 `GameDataLoader` 和 `KnowledgeBase` 摄入到 RAG 中。

- **RAG 存储** 位于 `rag_storage/` 和 `demo/bak/01/rag_storage/`：
   - `kb_registry.json` 映射游戏到 `game_puzzle1`, `game_puzzle2` 等。
   - 每个游戏的子目录（`game_puzzle1`, `game_puzzle2`），存储向量数据库、图和 KV 存储。

### 3.3 游戏流程（简化版）

1. 玩家开始一个新的游戏会话并选择一个谜题（或随机谜题）。
2. 系统创建带有相关 `KnowledgeBase` 和 `AgentGraph` 的 `GameSession`。
3. DM 介绍谜题陈述（使用 RAG / 结构化谜题文本）。
4. 玩家提问。
5. DM 以“裁判”身份评估问题，查阅隐藏答案和 RAG 存储，然后回应：
   - `YES`（是）、`NO`（否）、`YES AND NO`（是也不是）、`IRRELEVANT`（无关），或有限的自然语言解释，取决于游戏规则。
6. 游戏继续，直到玩家提出最终假设。
7. DM 根据标准答案评估假设并返回裁决。
8. DM 揭示完整解释、可选提示并记录完成情况。

---

## 4. RAG 子系统集成

RAG 子系统已经实现并用于谜题知识。本设计将其集成如下。

### 4.1 现有 RAG 组件（摘要）

- `KnowledgeBase` (`src/rag/knowledge_base.py`)
   - 管理用于谜题/场景游戏的多个知识库。
   - 使用基础存储目录下的 `_REGISTRY_FILE = "kb_registry.json"`。
   - 每个知识库都有一个 `KnowledgeBaseConfig`，包含 `kb_id`, `name`, `description`, `working_dir`, `provider_type`, `metadata`, `status` 等字段。
   - 提供异步方法：
     - `create_knowledge_base`, `delete_knowledge_base`。
     - `insert_documents(kb_id, documents)`。
     - `query(kb_id, query, **kwargs)`。
     - `health_check(kb_id)`, `close_all()`。

- `RAGProviderFactory` (`src/rag/provider_factory.py`)
   - 将提供商名称（`"lightrag"`, `"minirag"`）映射到提供商类。
   - 从 `RAGConfig` 创建提供商实例（`LightRAGProvider`, `MiniRAGProvider`）。

- `LightRAGProvider` 和 `MiniRAGProvider`
   - 用通用的 `BaseRAGProvider` 接口包装不同的 RAG 引擎（LightRAG, MiniRAG）。
   - 支持异步 `ainitialize`, `ainsert`, `aquery`, `astream_query`, `ahealth_check`。
   - 处理 LLM 和嵌入后端的配置（包括 Ollama 和 OpenAI 风格的 API）。

- `GameDataLoader` (`src/rag/tools/kb_loader.py`)
   - 将谜题 JSON 转换为 `RAGDocument`。
   - 为谜题陈述、答案、提示、附加信息等创建文档。

### 4.2 映射游戏到知识库

每个谜题/游戏都映射到一个隔离的 RAG 知识库：

- `kb_id` 命名约定：`game_<puzzle_id>`（例如 `game_puzzle1`）。
- `kb_registry.json` 存储元数据，包括：
   - `game_id`, `game_type`, `source_files`, `document_count`。

在 **游戏会话创建时**，引擎将：

1. 查找或为所选谜题创建知识库：
   - 首次运行时：使用 `GameDataLoader` 加载谜题 JSON -> `RAGDocument`。
   - 调用 `KnowledgeBase.create_knowledge_base(...)` 然后 `insert_documents(...)`。
2. 在 `GameSession` 中存储 `kb_id`。
3. 当 Agent 需要谜题相关信息时使用 `KnowledgeBase.query(kb_id, query, **kwargs)`。

### 4.3 Agent 的 RAG 使用

- **DM Agent**
   - 使用 RAG 验证玩家问题是否触及解决方案的相关部分。
   - 使用隐藏答案文档（标记为例如 `"type": "puzzle_answer"`）来确定真实性。
   - 可以检索支持上下文以获得更丰富的自然语言解释。

- **Player Agent（AI 控制）**
   - 可以使用 RAG 回忆以前看到的事实或谜题描述，而不是隐藏答案。
   - 访问应受到限制，以便 Player Agent 只能看到人类玩家会知道的内容。

### 4.4 RAG 中的访问控制

为了防止解决方案泄露：

- 知识库中的文档通过 `metadata` 标记：
   - `"type": "puzzle_statement"` – 公开。
   - `"type": "puzzle_answer"` – 秘密（仅限 Judge/DM）。
   - `"type": "hint"` – DM 控制。
   - `"type": "additional_info"` – 仅限 DM/Judge，可能会转化为提示。

- 暴露给每个 Agent 的 RAG 查询接口将应用 **元数据过滤器**：
   - Player 的 RAG 工具仅查询 `type in {puzzle_statement, public_fact}`。
   - Judge 的 RAG 工具查询所有类型，包括 `puzzle_answer`。
   - DM 的 RAG 工具可以查询提示和额外信息。

这是通过 **包装的 RAG 工具** 实现的，这些工具在内部调用带有适当过滤器的 `KnowledgeBase.query(...)`。

---

## 5. 记忆与 LangChain / LangGraph 集成设计

本系统的“记忆”分为**短期对话记忆**和**长期语义/情节记忆**两层，目标是：

- 会话内：让 Agent 能够理解当前局游戏中已经发生的事情（问题、裁决、提示等）。
- 跨会话：让系统能够记住玩家的画像、表现和全局统计，以支持个性化和战术调整。

设计上优先复用 **LangGraph 提供的长期记忆机制** 和 **LangChain 的短期对话 Memory 组件**，而不是完全自建存储层。

### 5.1 记忆层次与类型

结合 LangChain 文档中对记忆类型的划分（语义记忆 Semantic、情节记忆 Episodic 等），本系统的记忆分为三类，并与 LangChain / LangGraph 组件进行映射：

1. **会话（情节）记忆 – 短期 / 线程级**

    - 与单个 `GameSession` 紧密绑定，对应某一局海龟汤游戏内发生的所有事件。
    - 主要用途：
       - 为 DM / Player Agent 提供“本局发生了什么”的上下文（问题、回答、裁决、提示使用等）。
       - 支持 DM 进行回顾、总结以及根据回合历史调整判定。
    - 与 LangChain / LangGraph 的映射：
       - **LangGraph 图状态（State）+ Checkpoint**：
          - 每个会话对应一个 LangGraph 线程（`thread_id = session_id`）。
          - 图状态中包含 `turn_history`、`game_phase`、`hint_count` 等字段。
          - 使用 LangGraph 提供的 checkpoint 机制（例如 `MemorySaver` 或自定义 checkpoint 存储器）在每一步自动持久化状态，实现“线程级”短期记忆。
       - **LangChain 短期对话 Memory（可选）**：
          - 在具体 DM / Player Agent 节点内部，如果需要将最近 N 轮对话作为 Chat History 注入到提示中，可使用：
             - `ConversationBufferWindowMemory` 或
             - 基于 `BaseChatMessageHistory` 的自定义实现。
          - 这些组件从 LangGraph 的状态 / 事件日志中构造 `messages` 列表，并在调用 LLM 时作为上下文传入。

2. **玩家长期记忆（跨会话） – 语义记忆 Semantic Memory**

    - 与 `PlayerProfile` 关联，存储玩家在多局游戏中的聚合画像与表现统计：
       - 偏好的提问风格、常见错误模式、风险偏好。
       - 在不同谜题上的平均提问数、成功率、放弃率等。
    - 主要用途：
       - DM Agent 在 system prompt 中接入玩家画像，实现个性化叙述和难度调整。
       - Hint 策略可以根据玩家过往表现自动调整（更保守/更直接）。
    - 与 LangChain / LangGraph 的映射：
       - 使用 **LangGraph 的 `Store`** 作为长期语义记忆存储：
          - 每个玩家一个命名空间，例如：`namespace = "player:<player_id>"`。
          - 存储若干“记忆文档”（Memory Documents）：
             - `id`：文档键（如 `profile`, `performance_2025Q1`）。
             - `content`：玩家画像或表现的自然语言 / 结构化摘要。
             - `metadata`：`{"summary_type": "style_profile" | "performance_summary", "last_updated": ...}`。
          - 利用 Store 内置的**语义搜索**和**内容过滤**能力（`query` + `filter`），在 DM / Judge 节点前检索与当前谜题/场景最相关的玩家记忆，将其注入到系统提示中。

3. **全局记忆（系统级） – 语义 / 程序性记忆**

    - 与具体玩家和会话无关，描述系统整体层面的经验与策略：
       - 哪类提示策略更容易提升解谜体验。
       - 某些谜题整体难度统计、常见误区列表。
       - 可选：系统级“策略规则”（可以类比人类的 Procedural Memory）。
    - 与 LangChain / LangGraph 的映射：
       - 同样使用 **LangGraph Store**，但采用全局命名空间，例如：`namespace = "global"`。
       - 存储的文档示例：
          - `{ "summary_type": "hint_strategy", "content": "对于新手玩家，前 3 次提示尽量给出模糊线索……", "last_updated": ... }`。
          - `{ "summary_type": "puzzle_stats", "puzzle_id": "puzzle1", "content": "平均问题数 18，正确率 35%" }`。
       - DM / 系统管理工具可以周期性地汇总日志生成这些文档，为日后分析和调优提供基础。

### 5.2 数据模型与存储选型

#### 5.2.1 会话事件（Episodic Memory）

会话事件既会存在于 LangGraph 状态（便于在线推理），也可以追加写入 LangGraph Store（便于离线分析和跨会话汇总）。典型字段：

- `session_id`
- `turn_index`
- `timestamp`
- `role`：`DM` / `Judge` / `Player` / `System`
- `message`：自然语言内容（问题、回答、叙述等）
- `tags`：例如 `question`, `answer`, `hint`, `hypothesis`, `final_verdict`
- `raw_tool_calls`：可选字段，用于记录本回合触发的工具调用（方便调试和回放）

在存储层：

- LangGraph 状态：`turn_history` 作为一个按时间排序的事件数组，随图执行自动持久化到 checkpoint。
- LangGraph Store：命名空间 `session:<session_id>` 下追加写入事件文档，以支持后续的语义检索或离线聚合。

#### 5.2.2 玩家记忆记录（Semantic Memory）

玩家记忆记录主要以“摘要文档”的形式存在于 LangGraph Store 中：

- `player_id`
- `summary_type`：例如 `style_profile`, `performance_summary`
- `content`：自然语言摘要（可以包含结构化信息，例如 JSON 字符串）。
- `embedding`：由底层 Store 或向量后端管理（用于语义检索）。
- `last_updated`

当一局游戏结束或达成关键里程碑时，系统会：

1. 从 `session:<session_id>` 命名空间中读取该局的关键事件。
2. 调用 LLM 生成简洁的玩家表现 / 画像摘要。
3. 将生成的摘要以文档形式写入 `player:<player_id>` 命名空间（新增或合并 existing 记录）。

#### 5.2.3 存储实现和与 RAG 的关系

- **首选方案：LangGraph 自带 Store**
   - 使用 LangGraph 官方的 `Store` 抽象来管理记忆文档，利用其对语义检索和过滤的原生支持。
   - 优点：与 LangGraph 的长期 Agent / durable execution 机制深度集成，无需自行维护额外的索引结构。

- **高级方案：自定义 Store 后端复用现有 RAG 基础设施**
   - 如果希望底层只保留一套向量数据库 / KV 存储，可以实现一个自定义的 `Store` 后端：
      - 在内部将 `Store.put` / `Store.search` 等操作映射到现有 `KnowledgeBase` / LightRAG / MiniRAG 的底层向量存储 API。
   - 这样，谜题知识（`kb_id = game_puzzle1` 等）和长期记忆可以共用底层技术栈，但在逻辑命名空间上保持严格隔离。

### 5.3 记忆相关操作与工具化

记忆操作会在 LangChain / LangGraph 中作为**工具（Tool）或专门的图节点**出现，并融入到游戏流程中：

1. `append_session_event(session_id, event)`

    - 角色：在每一回合后被调用。
    - 动作：
       - 将 `event` 追加到 LangGraph 状态中的 `turn_history`。
       - （可选）将 `event` 作为文档写入 `session:<session_id>` 命名空间的 Store，用于后续汇总和语义检索。

2. `get_session_history(session_id, limit=N)`

    - 角色：为 DM / Player Agent 提供最近 N 条对话上下文。
    - 动作：
       - 直接从当前线程的 LangGraph 状态中获取 `turn_history` 的尾部 N 条。
       - 如需更长历史，可从 `session:<session_id>` Store 命名空间中按时间倒序读取。

3. `summarize_session(session_id)`

    - 角色：在一局游戏结束时调用，用于生成本局摘要和更新玩家长期记忆。
    - 动作：
       1. 从 LangGraph 状态 / Store 中获取本局关键事件。
       2. 调用 LLM 生成自然语言摘要：包括玩家解题思路、常见误区、表现亮点等。
       3. 以文档形式写入 `player:<player_id>` 命名空间，并视情况写入 `global` 命名空间（例如更新难度统计）。

4. `update_player_profile(player_id, new_info)`

    - 角色：在有新的玩家摘要或重要信息时调用。
    - 动作：
       - 从 `player:<player_id>` 命名空间中检索现有画像文档（按 `summary_type` 或语义检索）。
       - 调用 LLM 合并旧画像与 `new_info`，生成新的统一画像。
       - 将结果覆盖写回 `profile` 文档。

5. `retrieve_player_profile(player_id)`

    - 角色：在 DM / Judge Agent 处理玩家输入之前调用。
    - 动作：
       - 对 `player:<player_id>` 命名空间执行语义检索，选出 1–3 条最相关的画像/表现摘要。
       - 将这些摘要作为系统提示的一部分注入 DM / Judge 的上下文中，以影响语气、提示策略等。

这些操作将通过 LangChain 的 Tool 机制和 LangGraph 的节点封装，对上层游戏逻辑暴露为简单的 API，而在内部充分利用 LangGraph 的长期记忆与 LangChain 的对话记忆能力。

---

## 6. 模型提供商抽象

系统必须支持使用 **Ollama** 或 **基于 API 的模型**，且对游戏逻辑的更改最小。

### 6.1 LLM 和嵌入客户端

定义抽象接口（伪契约）：

- `LLMClient`
   - `generate(prompt, **params) -> str` (同步或异步变体)。
   - `stream(prompt, **params) -> AsyncIterator[str]`。

- `EmbeddingClient`
   - `embed(texts: List[str]) -> List[List[float]]`。

实现类：

1. **OllamaLLMClient / OllamaEmbeddingClient**
   - 使用本地 Ollama HTTP 端点。
   - 可配置基础 URL、模型名称、温度等。

2. **OpenAICompatibleLLMClient / OpenAICompatibleEmbeddingClient**
   - 使用 OpenAI / Azure / 其他 API 兼容服务。
   - 可配置基础 URL、API 密钥、模型名称、额外参数。

这些客户端被以下组件使用：

- 现有的 RAG 提供商（`LightRAGProvider`, `MiniRAGProvider`）通过其配置选项。
- Agent 层中新的 LangChain `ChatModel` 和 `Embeddings` 包装器。

### 6.2 提供商选择

中心配置（YAML/JSON，例如 `config/model_providers.json`）：

- 示例设置：

   - `provider: "ollama" | "api"`。
   - `llm_model_name`, `embedding_model_name`。
   - `api_key`, `base_url`。
   - `default_temperature`, `max_tokens` 等。

启动时：

- `ModelProviderRegistry` 读取配置并返回实例化的 `LLMClient` 和 `EmbeddingClient`。
- RAG 组件通过 `KnowledgeBase` / 提供商选项相应配置。
- LangChain 模型由这些低级客户端构建。

---

## 7. 使用 LangChain 和 LangGraph 进行编排

### 7.1 整体图结构

使用 LangGraph 定义 **游戏状态机** 和 Agent 之间的交互。

推荐的顶层图：

- 入口节点：`PlayerMessageNode`（从 UI 接收用户输入）。
- 基于消息类型（问题、假设、命令）的路由逻辑。
- 子图：
   - **提问流程** – 玩家提问。
   - **假设流程** – 玩家提出解决方案。
   - **提示流程** – 玩家显式或隐式请求提示。

每个流程涉及特定角色：

- **DM 节点** – 用于叙事、说明、回顾以及对问题/假设做出规则裁决。
- **RAG 工具节点** – 用于知识检索。
- **记忆更新节点** – 用于持久化事件和更新摘要。

### 7.2 Agent 定义（概念性）

#### DM Agent

职责：

- 介绍谜题。
- 给出不泄露答案的澄清。
- 提供故事风味和沉浸感。
- 将玩家输入分类为 **问题**、**假设**、**元命令**（例如 `restart`, `hint`）等。
- 对于问题：回答 `YES`（是） / `NO`（不是） / `YES AND NO`（是也不是） / `IRRELEVANT`（无关） 和可选的简短评论。
- 对于假设：根据标准答案判断是否正确或部分正确，并决定游戏是否结束。

输入：

- 谜题元数据和完整 RAG 上下文（包括答案和提示）。
- 会话历史（最近 N 回合）和玩家长期记忆。

工具：

- `get_full_puzzle_context(kb_id)` – 查询所有文档类型的 RAG。
- `append_session_event`。
- `update_session_state`。

#### Player Agent（可选 AI 玩家）

在某些模式下，可以使用 AI Player Agent 自动玩谜题。

职责：

- 生成好问题。
- 根据 DM 的裁决和提示完善假设。

输入：

- 公开谜题陈述和对话历史。
- （可选）对公开文档的有限 RAG 访问。

工具：

- `get_public_puzzle_context(kb_id)`。
- `append_session_event`。

### 7.3 LangGraph 中的状态和记忆

定义一个 **图状态对象**（用于 LangGraph），包含如下字段：

- `session_id`
- `kb_id`
- `player_id`
- `turn_index`
- `last_user_message`
- `message_type` (`question`, `hypothesis`, `command`)
- `last_dm_response`
- `game_phase` (`intro`, `playing`, `awaiting_final`, `completed`)
- `hint_count`
- `score`

状态转换由节点和条件边控制。例如：

1. `PlayerMessageNode` 更新 `last_user_message` 并调用 **分类器** (LLM) 设置 `message_type`。
2. 如果 `message_type == "question"` -> 发送到 `JudgeNode`。
3. 如果 `message_type == "hypothesis"` -> 发送到 `JudgeHypothesisNode`。
4. 在每个节点之后，调用 `MemoryUpdateNode` 持久化事件。
5. 如果 `game_phase` 变为 `completed`，路由到 `RevealSolutionNode`。

### 7.4 LangChain 中的工具

将关键操作暴露为工具：

- `rag_query_public(kb_id, query, **filters)`。
- `rag_query_full(kb_id, query, **filters)`。
- `append_event(session_id, event)`。
- `get_recent_events(session_id, limit)`。
- `summarize_session(session_id)`。
- `get_player_profile(player_id)`。
- `update_player_profile(player_id, summary)`。

这些工具由在 LangChain 中定义为 `Tool‑using LLM Chains` 或 `Agents` 的 Agent 使用，然后嵌入到 LangGraph 节点中。

---

## 8. 游戏引擎和会话管理

### 8.1 GameEngine 职责

- 初始化：
   - 加载配置（模型提供商、RAG 提供商、基础目录）。
   - 使用 `base_storage_dir`（例如 `rag_storage/`）初始化 `KnowledgeBase`。
   - 初始化 `ModelProviderRegistry` 和 LangChain 模型。

- 游戏生命周期：
   - 使用 `GameDataLoader.discover_games(...)` 发现可用谜题。
   - 创建新会话和关联的 `kb_id`。
   - 管理会话查找（通过 `session_id`）和持久化。

- API 表面（概念性）：
   - `list_puzzles() -> List[PuzzleSummary]`。
   - `create_session(puzzle_id, player_id, **options) -> GameSession`。
   - `handle_message(session_id, player_message) -> system_response`。
   - `get_session_state(session_id) -> SessionState`。

### 8.2 GameSession 职责

- 持有运行时状态和引用：
   - `AgentGraph` 实例。
   - `KnowledgeBase` / `kb_id`。
   - `MemoryManager`。
   - `PlayerProfile`。

- 使用持久状态对象编排对 LangGraph 的调用。

- 提供方法：
   - `process_player_input(message: str) -> str | RichResponse`。
   - `save()` – 持久化状态和记忆快照。
   - `load(...)` – 从存储重建。

### 8.3 错误处理和健康检查

- 在启动会话之前，引擎可以调用 `KnowledgeBase.health_check(kb_id)` 以确保 RAG 提供商可用。
- 如果提供商不可用，系统可以：
   - 回退到更简单的模式（无 RAG，仅静态谜题文本）。
   - 向用户显示友好的错误。

---

## 9. 存储和持久化

### 9.1 现有 RAG 存储

结构如下：

- `rag_storage/kb_registry.json` – 知识库注册表。
- `rag_storage/game_puzzleX/` – 每个 KB 的存储，包含：
   - `graph_chunk_entity_relation.graphml`
   - `kv_store_*.json`
   - `vdb_*.json`

新设计继续使用此结构，并将所有 RAG 相关的持久化委托给现有的 `KnowledgeBase` 和提供商实现。

### 9.2 会话和记忆的新存储

建议目录：

- `game_storage/sessions/` – 每个 `session_id` 的序列化会话状态。
   - 例如，`sessions/<session_id>.json` 包含高层状态（阶段、分数、活动谜题等）。

- `game_storage/events/` – 仅追加日志。
   - 例如，`events/<session_id>.jsonl`，每个事件一个 JSON 对象。

- `game_storage/player_memory/` – 玩家长期记忆。
   - 例如，`player_memory/<player_id>.json` 包含聚合摘要。

可选地，使用与 RAG 相同的向量数据库基础设施进行语义记忆，使用专用的 `kb_id` 前缀（例如 `memory_player_<player_id>`）。

---

## 10. 配置和可扩展性

### 10.1 配置文件

创建一个专用的配置目录，例如 `config/`，包含：

- `config/game.yaml`
   - 默认 RAG 提供商（`lightrag` / `minirag`）。
   - 数据和存储的基础目录。
   - 默认模型和参数。

- `config/models.yaml`
   - 提供商选择（`ollama` 或 `api`）。
   - 模型名称、端点、API 密钥（如果敏感则不放入 VCS）。

- `config/agents.yaml`
   - 每个角色的提示和行为设置：
     - DM 人设（语气、风格）。
     - Judge 严格程度、允许的冗长程度。
     - 提示策略（何时给出提示，提示强度）。

### 10.2 扩展系统

- **新 RAG 提供商**
   - 实现 `BaseRAGProvider` 子类。
   - 通过 `RAGProviderFactory.register_provider(name, cls)` 注册。

- **新模型后端**
   - 为新后端实现 `LLMClient` / `EmbeddingClient`。
   - 扩展 `ModelProviderRegistry` 以识别并实例化它。

- **新 Agent 角色**
   - 定义角色的提示和工具。
   - 添加 LangGraph 节点和边以将其集成到流程中。

- **新游戏类型（例如，谋杀之谜）**
   - 实现新的 `GameDataLoader` 变体或扩展现有的。
   - 定义新文档元数据类型和 RAG 访问规则。
   - 调整游戏流程图以反映新类型。

---

## 11. 安全和安全注意事项

- **信息泄露预防**
   - 通过 RAG 过滤器严格分离公开与秘密文档。
   - 仔细的 Judge/DM 系统提示，指示它们除非在最后，否则不要泄露完整答案。

- **提示注入缓解**
   - 清理来自玩家的输入并将其视为不可信。
   - 避免将玩家消息直接连接到工具控制提示中。

- **资源限制**
   - 可配置每会话的最大回合数和提示数。
   - 每会话令牌预算和速率限制。

---

## 12. 示例端到端流程（叙事）

1. **启动**
   - 加载配置并初始化模型提供商。
   - 使用基础目录 `rag_storage/` 初始化 `KnowledgeBase`。

2. **谜题准备**
   - 使用 `GameDataLoader.discover_games` 发现 `data/situation_puzzles/` 下的谜题。
   - 对于每个选定的谜题：
     - 确保存在相应的 `kb_id`（如果丢失则创建 + 摄入）。

3. **创建会话**
   - 玩家选择 `puzzle1`。
   - 引擎创建 `GameSession`，包含：
     - `kb_id = game_puzzle1`。
     - 一个新的 `session_id`。
     - 一个带有 `phase = intro` 的 LangGraph 状态对象。

4. **介绍阶段**
   - DM Agent 使用 `rag_query_public` 获取谜题陈述并进行叙述。
   - 更新会话历史和记忆。

5. **提问阶段**
   - 玩家提交问题。
   - 图将消息路由到 DM Agent。
   - DM Agent 使用 `rag_query_full` 检查答案和相关事实，然后返回 `YES` / `NO` / `YES AND NO` / `IRRELEVANT` 以及可选解释。
   - 更新记忆和状态（回合计数器、提示使用）。

6. **假设阶段**
   - 玩家陈述完整假设。
   - DM Agent 通过 RAG 将其与标准答案进行比较。
   - 如果足够正确，`phase` 转换为 `completed`。

7. **解决阶段**
   - DM Agent 揭示完整答案和遗漏的细节。
   - 系统总结会话并更新 `PlayerProfile` 长期记忆。

---

## 13. 实施路线图（高层）

1. **基础**
   - 引入配置文件（`config/`）、`ModelProviderRegistry` 和记忆存储文件夹结构。
   - 将现有的 `KnowledgeBase` 和 `GameDataLoader` 包装在 `KnowledgeBaseManager` 服务中以供游戏使用。

2. **游戏引擎和会话核心**
   - 实现 `GameEngine`, `GameSession`, `PuzzleRepository`, `MemoryManager`（最初仅结构代码）。
   - 将引擎连接到 RAG 子系统和模型提供商。

3. **LangGraph Agent 编排**
   - 定义图状态模式。
   - 实现基本的 DM 节点以及简单的提问流程。
   - 集成记忆更新工具。

4. **接口层**
   - 用于开始游戏和与会话交互的 CLI 或最小 Web UI。

5. **长期记忆增强**
   - 添加会话摘要和玩家画像更新。

6. **高级功能**
   - AI Player Agent。
   - 战役模式和多个同时进行的会话。
   - 来自游戏日志的分析仪表板。

本设计保留了现有的 RAG 实现作为谜题知识的基础，同时使用 LangChain 和 LangGraph 在其之上添加了一个结构化的多 Agent、多提供商、具有记忆感知的游戏系统。
