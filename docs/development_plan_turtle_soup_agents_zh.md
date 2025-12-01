# Echoes of Deceit – 多 Agent 海龟汤系统开发计划

## 0. 范围与约束

本开发计划将中文系统设计文档转化为具体的、**分阶段实施的路线图**，用于构建多 Agent “海龟汤”（情境猜谜）游戏系统。本计划面向熟悉 **LangChain**、**LangGraph** 和 **RAG** 模式的 Python 后端团队。

**重要约束：**

- 计划必须**完全符合提供的系统设计**（架构、数据流和职责）。
- 计划**绝不能包含具体代码**（无函数体，无完整代码清单）。应侧重于模块、职责、API 和配置。
- 计划假设：
  - Python 3.12+。
  - 现有的 RAG 子系统位于 `src/rag/`，如设计所述（`KnowledgeBase`、提供商、`GameDataLoader`）。
  - 未来使用 LangChain 和 LangGraph 进行编排和记忆管理。

计划分为多个**阶段**。每个阶段都可以增量交付，并应包含基本测试和手动验证。

---

## 第一阶段 – 基础、配置与 RAG 集成加固

### 1.1 目标

建立坚实的基础：配置管理、模型提供商抽象，以及在现有 RAG 子系统之上的轻量级 **KnowledgeBaseManager**。暂不引入 LangGraph；重点是**清晰地访问谜题知识**。

### 1.2 交付物

- `config/` 下的中心配置文件。
- 模型提供商抽象（`LLMClient`、`EmbeddingClient`、`ModelProviderRegistry`）。
- 包装 `KnowledgeBase` 的 `KnowledgeBaseManager` 服务。
- 初始 CLI 或简单脚本，用于：
  - 发现 `data/situation_puzzles/` 下的谜题。
  - 确保存储在 `rag_storage/` 下的相应知识库存在。
  - 运行基本健康检查。

### 1.3 主要任务

#### 1.3.1 配置布局

创建一个 `config/` 目录，至少包含：

- `config/game.yaml`
  - 默认 RAG 提供商（例如 `lightrag` 或 `minirag`）。
  - 基础目录：
    - `data_base_dir` -> `data/situation_puzzles/`。
    - `rag_storage_dir` -> `rag_storage/`（以及可能的 `demo/bak/` 以保持兼容性）。
    - 未来的 `game_storage_dir` -> `game_storage/`。
  - 默认语言、默认谜题集和游戏级参数（例如最大回合数、默认提示限制）。

- `config/models.yaml`
  - `provider: "ollama" | "api"`。
  - `llm_model_name` 和 `embedding_model_name`。
  - `api_base_url`、`api_key` 或环境变量引用（不要在版本控制中存储真实密钥）。
  - 默认生成参数（温度、最大 token 数等）。

- `config/agents.yaml`
  - DM 人设（语气、风格、关于不提前泄露答案的严格程度）。
  - 裁判（Judge）约束（简洁/详细程度，如何措辞 是/否 回答）。
  - 提示策略：何时以及如何提供提示的规则。

**实施说明（无代码）：**

- 使用 YAML 解析库在启动时将这些配置加载到内存配置对象中。
- 定义清晰的数据结构来表示这些配置，以便其他层（引擎、Agent、RAG 提供商）可以使用它们。

#### 1.3.2 模型提供商抽象

引入**模型提供商层**，隐藏 Ollama 与 API 提供商的细节。

**接口（概念性）：**

- `LLMClient`
  - `generate(prompt, **params) -> str`（同步或异步变体，视选择而定）。
  - `stream(prompt, **params) -> iterator/async iterator of str` 用于流式输出。

- `EmbeddingClient`
  - `embed(texts: List[str]) -> List[List[float]]`。

**具体实现：**

- `OllamaLLMClient` / `OllamaEmbeddingClient`
  - 使用 Ollama HTTP 端点和 `config/models.yaml` 中的配置。

- `OpenAICompatibleLLMClient` / `OpenAICompatibleEmbeddingClient`
  - 根据配置使用 OpenAI、Azure OpenAI 或兼容的 API。

**ModelProviderRegistry（模型提供商注册表）：**

- 读取 `config/models.yaml`。
- 根据 `provider` 字段，实例化相应的客户端。
- 为系统的其余部分提供单一入口点：
  - `get_llm_client()`。
  - `get_embedding_client()`。

**参考文档：**

- LangChain 模型文档（例如 `ChatOpenAI`、`ChatOllama`），重点关注它们如何包装基础 LLM 客户端。
- 具体提供商的 HTTP API 文档，了解具体的请求格式。

#### 1.3.3 `KnowledgeBaseManager` 服务

包装现有的 `KnowledgeBase` 类，使游戏代码不直接操作提供商细节。

职责：

- 使用配置中的 `rag_storage_dir` 和提供商设置初始化 `KnowledgeBase`。
- 暴露面向领域的方法：
  - `ensure_puzzle_kb(puzzle_id) -> kb_id`。
  - `query_public(kb_id, query, filters)`（例如，仅 `puzzle_statement` 和公开事实）。
  - `query_full(kb_id, query, filters)`（包括 `puzzle_answer`、提示等）。
  - `health_check(kb_id)`。
- 使用 `GameDataLoader` 来：
  - 发现 `data/situation_puzzles/` 中的游戏/谜题。
  - 将谜题 JSON 转换为 `RAGDocument` 实例。
  - 使用 `KnowledgeBase.insert_documents` 将它们插入到适当的知识库中。

RAG **访问控制**必须符合设计：

- 在文档中使用 `metadata.type` 标记：
  - `puzzle_statement`（公开），
  - `puzzle_answer`（秘密），
  - `hint`（受控），
  - `additional_info`（仅限 DM/Judge）。
- 实现过滤逻辑，使得：
  - Player Agent 仅查询公开类型。
  - DM/Judge Agent 可以查询完整上下文。

#### 1.3.4 CLI / 实用脚本

创建一个最小的 CLI 或主脚本（暂无复杂 I/O），能够：

- 加载配置。
- 初始化 `ModelProviderRegistry` 和 `KnowledgeBaseManager`。
- 发现谜题并确保知识库存在：
  - 对于每个谜题，创建或更新知识库 `game_<puzzle_id>`。
- 对每个知识库运行 `health_check` 并打印状态报告。

此阶段暂无 LangGraph。目的是验证：

- 模型配置工作正常。
- RAG 管道和元数据标记正确。

---

## 第二阶段 – 游戏领域模型与存储层

### 2.1 目标

定义并持久化**游戏领域实体**（`Puzzle`、`Game`、`GameSession`、`PlayerProfile` 等）和基于文件的基础存储。仍然极少或没有 LangGraph；为后续编排准备坚实的领域模型。

### 2.2 交付物

- 所有核心实体的 Python 领域模型类（或 dataclasses）。
- `game_storage/` 下基于文件的存储布局。
- 用于创建、加载和更新会话及玩家画像的辅助 API。

### 2.3 主要任务

#### 2.3.1 领域实体

将领域实体定义为 Python 类（除了简单的验证和辅助函数外，不包含业务逻辑）：

- `Puzzle`
  - `id`、`title`、`description`。
  - `puzzle_statement`、`answer`。
  - `constraints`（例如最大提问数、语言规则）。
  - `tags`。

- `Game`
  - 映射到一个或多个 `Puzzle` 对象（简单版本：与谜题 1:1）。

- `GameSession`
  - `session_id`（UUID 或类似）。
  - `puzzle_id`（字符串）。
  - `player_ids` 或 `PlayerProfile` 引用。
  - `state`：`LOBBY`（大厅）、`IN_PROGRESS`（进行中）、`COMPLETED`（已完成）、`ABORTED`（已中止）。
  - `turn_history`（简单的结构化事件；见稍后的记忆部分）。
  - `config`（会话创建时拍摄的 LLM、RAG、记忆设置快照）。

- `PlayerProfile`
  - `player_id`。
  - `display_name`。
  - `preferences`（难度、语言等）。
  - `long_term_memory_ref`（指向记忆命名空间或文件路径的指针）。

- `AgentRole`
  - 枚举或类似：`DM`、`Player`、`Observer`、`HintMaster`（未来）。

确保这些模型定义独立于 LangChain/LangGraph 类型，纯粹关注领域。

#### 2.3.2 谜题仓库 (Puzzle Repository)

创建一个 `PuzzleRepository` 服务，用于：

- 从 `data/situation_puzzles/` 读取谜题 JSON。
- 将原始 JSON 字段映射到 `Puzzle` 对象。
- 提供 API：
  - `list_puzzles() -> List[PuzzleSummary]`（id、标题、难度、标签）。
  - `get_puzzle(puzzle_id) -> Puzzle`。
  - `find_random_puzzle(filters) -> Puzzle`。

在内部，它应尽可能复用 `GameDataLoader` 的发现逻辑，但保持仓库接口独立于 RAG 细节。

#### 2.3.3 游戏存储布局

遵循设计，在 `game_storage/` 下创建目录布局：

- `game_storage/sessions/`
  - `sessions/<session_id>.json` 用于序列化的 `GameSession` 状态。

- `game_storage/events/`
  - `events/<session_id>.jsonl` 用于仅追加的事件日志（每行一个 JSON 事件）。

- `game_storage/player_memory/`
  - `player_memory/<player_id>.json` 用于聚合的玩家摘要/画像。

定义**序列化和反序列化规则**：

- 实体如何映射到 JSON（无循环引用；使用 ID 表示关系）。
- 枚举/状态如何存储（例如字符串 `"IN_PROGRESS"`）。
- 用于潜在模式演变的向后兼容版本字段（例如 `schema_version`）。

#### 2.3.4 会话管理 API

定义一个 `GameSessionStore` 或类似的管理器，包含如下方法：

- `create_session(puzzle_id, player_ids, options) -> GameSession`。
- `save_session(session: GameSession)`。
- `load_session(session_id) -> GameSession`。
- `list_sessions(filters) -> List[GameSession]`。

此存储在此阶段应基于文件，稍后可插拔数据库后端。

#### 2.3.5 基本测试和手动流程

- 最小单元测试：
  - `PuzzleRepository`（加载、列出、获取谜题）。
  - `GameSession` 序列化/反序列化。
  - `GameSessionStore` 读/写。

- 手动脚本：
  - 创建一个虚拟 `PlayerProfile`。
  - 为选定的谜题创建一个新会话。
  - 保存并重新加载会话。

---

## 第三阶段 – 记忆模型与 LangGraph Store 集成

### 3.1 目标

引入设计中描述的记忆层：

1. 每个游戏会话的**会话（情节）记忆**。
2. **玩家级长期语义记忆**。
3. **全局系统级记忆**。

在概念上将它们与 **LangGraph 的 `Store`** 抽象集成，同时保留基于文件的回退表示。

### 3.2 交付物

- 会话事件和玩家记忆记录的记忆模型定义。
- 抽象 `MemoryStore` 接口（尽可能与 LangGraph 的 `Store` 对齐）。
- 使用文件布局的具体实现，以及可选的与现有 RAG 向量存储集成的语义搜索。

### 3.3 主要任务

#### 3.3.1 会话事件模式（情节记忆）

定义标准的事件记录结构，例如：

- `session_id`。
- `turn_index`。
- `timestamp`。
- `role`：`DM` | `Player` | `Judge` | `System`。
- `message`（问题/回答/叙述的文本内容）。
- `tags`：token 列表，如 `question`、`answer`、`hint`、`hypothesis`、`final_verdict`。
- `raw_tool_calls`（可选的结构化记录，记录该回合使用的 RAG / 记忆工具）。

映射：

- 存储在 LangGraph 状态（`turn_history` 字段）中用于在线使用。
- 序列化为 `.jsonl` 存储在 `game_storage/events/<session_id>.jsonl` 中用于离线分析。

#### 3.3.2 玩家记忆记录（语义记忆）

定义玩家记忆摘要为文档，稍后可存储在 LangGraph `Store` 或 RAG 后端：

- `player_id`。
- `summary_type`：例如 `style_profile`、`performance_summary`。
- `content`：自然语言文本（可能包含用于结构化统计的嵌入 JSON）。
- `metadata`：例如 `{"summary_type": ..., "last_updated": ...}`。

设计这些摘要的创建和更新方式：

- 在会话结束（或关键里程碑）时，触发摘要过程。
- 它读取会话事件并生成紧凑的、人类可读的摘要。
- 该摘要与现有的画像信息（如果有）合并并持久化。

#### 3.3.3 全局记忆记录

定义全局系统知识的文档：

- `summary_type`：例如 `hint_strategy`、`puzzle_stats`。
- `puzzle_id`（可选，用于特定谜题的统计）。
- `content`：总结所有玩家和会话中有效策略的自然语言。

这些文档位于 LangGraph `Store` 的全局命名空间（`global`）或专用的 `game_storage/global_memory/` 目录中。

#### 3.3.4 MemoryStore 抽象

定义一个小型的抽象，代表 LangGraph 风格的存储，但与具体实现解耦：

- 能力（概念性）：
  - `put(namespace, id, document, metadata)`。
  - `get(namespace, id)`。
  - `search(namespace, query, filters, k)`。

实施策略：

- **第三阶段基线：**
  - 基于文件的文档存储在 `game_storage/player_memory/` 和 `game_storage/global_memory/` 下。
  - 简单的搜索（例如，通过元数据过滤和/或朴素字符串匹配）以保持实现轻量。

- **未来升级：**
  - 实现一个 `Store` 后端，复用现有的 RAG 向量基础设施（LightRAG/MiniRAG）进行嵌入和语义搜索。

#### 3.3.5 记忆相关操作（逻辑 API）

将这些操作设计为**逻辑服务**（稍后暴露为 LangChain 工具 / LangGraph 节点）：

1. `append_session_event(session_id, event)`
   - 将事件添加到内存中的 `turn_history`。
   - 追加到 `events/<session_id>.jsonl`。

2. `get_session_history(session_id, limit)`
   - 从内存状态（或在需要时从文件）返回最新的 `N` 个事件。

3. `summarize_session(session_id)`
   - 读取事件。
   - 使用 LLM 创建会话和玩家行为的文本摘要。
   - 通过 `MemoryStore` 将摘要保存到 `player:<player_id>` 命名空间。
   - 可选地更新全局摘要（`global` 命名空间）。

4. `update_player_profile(player_id, new_summary)`
   - 检索玩家的现有画像文档。
   - 通过 LLM 或启发式规则与 `new_summary` 合并。
   - 写回更新后的画像文档。

5. `retrieve_player_profile(player_id)`
   - 搜索 `player:<player_id>` 命名空间。
   - 返回 1–3 个最相关的记忆文档，用于注入 DM/Judge 提示。

在此阶段，这些是纯服务；LangGraph 集成将在下一阶段进行。

**参考文档：**

- LangGraph `Store` 文档。
- LangChain 记忆类型（ConversationBuffer、语义记忆模式）用于概念对齐。

---

## 第四阶段 – 游戏引擎核心（无完整 LangGraph）

### 4.1 目标

构建 **GameEngine** 和核心 `GameSession` 运行时逻辑，使用领域模型、存储和 RAG 子系统，但**在引入完整的 LangGraph 状态机之前**。这允许在更简单的结构（例如同步函数调用）中进行早期端到端流程，并在 LangGraph 不可用时提供回退。

### 4.2 交付物

- 协调配置、RAG、谜题和会话的 `GameEngine` 服务。
- 提供高级方法（`process_player_input` 等）的 `GameSession` 运行时包装器。
- 基本的 CLI 循环，用于通过终端手动玩会话。

### 4.3 主要任务

#### 4.3.1 GameEngine 职责

实现一个高级 `GameEngine` 组件，其职责与设计一致：

- 初始化：
  - 加载配置（`game.yaml`、`models.yaml`、`agents.yaml`）。
  - 初始化 `ModelProviderRegistry`。
  - 初始化 `KnowledgeBaseManager`。
  - 初始化 `PuzzleRepository`。
  - 初始化存储（`GameSessionStore`、`MemoryStore`）。

- 游戏生命周期管理：
  - `list_puzzles() -> List[PuzzleSummary]`。
  - `create_session(puzzle_id, player_id, options) -> GameSession`。
  - `get_session(session_id) -> GameSession`。
  - `save_session(session_id)` 和 `load_session(session_id)` 包装器。

- 健康检查：
  - 对于给定的谜题/会话，调用 `KnowledgeBaseManager.health_check(kb_id)`。
  - 如果 RAG 不可用，提供人类可读的错误和回退策略（例如，降级为静态谜题文本模式）。

#### 4.3.2 GameSession 运行时逻辑（LangGraph 前）

此阶段的 `GameSession` 应：

- 持有引用：
  - `Puzzle`。
  - `kb_id`（来自 `KnowledgeBaseManager`）。
  - `PlayerProfile`。
  - `turn_history`（会话事件列表）。
  - 当前 `state` 和阶段（intro、playing、awaiting_final、completed）。

- 为接口层暴露单一方法：
  - `process_player_input(message: str) -> response object`。

在内部，此方法应：

1. 分类输入类型：
   - 使用简单的分类启发式或 LLM 来确定输入是否为：
     - 关于谜题的**问题**。
     - **最终假设**（尝试解答）。
     - **元命令**（例如，重新开始，请求提示）。

2. 路由到适当的内部处理程序：
   - 问题 -> DM/Judge 逻辑（在此阶段，可以是简化的基于规则的或引用完整谜题上下文的单个 LLM 提示）。
   - 假设 -> 将假设与答案进行比较的评估逻辑。
   - 命令 -> 处理重新开始、提示或状态查询。

3. 适当地更新 `turn_history` 和 `GameSession` 状态。

此阶段可以使用一个或多个 LLM 提示，但尚不需要完整的 LangGraph 节点/边定义。

#### 4.3.3 简化的 DM/Judge 逻辑

设计 DM/Judge 行为，使其大致符合最终期望的行为，但可以实现为**每步单个链式提示**：

- 在评估问题或假设时，使用 `KnowledgeBaseManager.query_full` 检索相关信息（包括隐藏答案）。
- 强制执行：
  - 对于问题：
    - 输出 `YES` / `NO` / `YES AND NO` / `IRRELEVANT` 加上可选的简短解释。
  - 对于假设：
    - 输出裁决（正确/不正确/部分正确）和可选解释。
- 使用 `agents.yaml` 配置实现可配置的冗长程度和风格（DM 人设）。

这提供了一个可以通过 CLI 测试的工作游戏循环。

---

## 第五阶段 – 基于 LangGraph 的编排与 Agent 图

### 5.1 目标

重构运行时以使用 **LangGraph** 进行有状态的多 Agent 编排，匹配设计的概念图：DM、Player、RAG 和记忆节点，具有明确的状态转换。

### 5.2 交付物

- 代表 `GameSession` 运行时状态的 LangGraph `State` 模式。
- 实现游戏流程的 Agent 图（节点和边）：
  - 问题处理。
  - 假设处理。
  - 提示处理。
- 集成记忆操作和 RAG 作为工具/节点。

### 5.3 主要任务

#### 5.3.1 LangGraph 的状态模式

定义一个 LangGraph **状态对象**来镜像或封装 `GameSession` 运行时状态：

- 字段（概念性）：
  - `session_id`。
  - `kb_id`。
  - `player_id`。
  - `turn_index`。
  - `last_user_message`。
  - `message_type`（`question`、`hypothesis`、`command`）。
  - `last_dm_response`。
  - `game_phase`（`intro`、`playing`、`awaiting_final`、`completed`）。
  - `hint_count`。
  - `score`。
  - `turn_history`（或对其的引用）。

确保此状态对象与 LangGraph 的检查点机制（`MemorySaver` 或自定义检查点存储）兼容，以便每一步都自动持久化。

#### 5.3.2 节点设计

定义 LangGraph 图中的概念节点：

- `PlayerMessageNode`
  - 来自 UI 的入口点。
  - 读取原始输入并填充 `last_user_message`。
  - 触发 `message_type` 的分类（通过规则或 LLM 调用）。

- `RouteByMessageTypeNode`
  - 基于 `message_type` 的条件路由器。
  - 将状态发送到 `QuestionFlow`、`HypothesisFlow` 或 `CommandFlow` 子图。

- `DMQuestionNode`
  - 实现 DM 对玩家问题的处理。
  - 通过 `rag_query_full` 工具调用 RAG。
  - 生成结构化的 DM 响应：裁决和解释。

- `DMHypothesisNode`
  - 使用 RAG（包括 `puzzle_answer`）将假设与答案进行比较。
  - 如果解决方案足够正确，将 `game_phase` 更新为 `completed`。

- `CommandHandlerNode`
  - 处理元命令（重新开始、提示请求、状态查询）。

- `HintNode`
  - 通过 RAG 受控访问提示文档。
  - 使用 `agents.yaml` 决定提示应该多强/多明确。

- `MemoryUpdateNode`
  - 在每次 DM/Player 交互后，追加一个会话事件。
  - 游戏完成后可选地触发总结。

每个节点应：

- 操作并更新共享状态对象。
- 使用 RAG、记忆和会话存储的工具，而不是直接处理它们。

#### 5.3.3 工具与集成

将领域服务暴露为可在 LangGraph 节点内使用的 LangChain **工具**：

- RAG 工具：
  - `rag_query_public(kb_id, query, filters)`。
  - `rag_query_full(kb_id, query, filters)`。

- 记忆工具：
  - `append_event(session_id, event)`。
  - `get_recent_events(session_id, limit)`。
  - `summarize_session(session_id)`。
  - `get_player_profile(player_id)`。
  - `update_player_profile(player_id, summary)`。

映射到 LangGraph：

- 每个工具变为：
  - 通过 LangChain 工具接口包装的可调用对象，并嵌入到 LLM 提示中。
  - 或者一个独立的 LangGraph 节点，执行其效果而无需 LLM 推理。

#### 5.3.4 检查点与持久化

配置 LangGraph 以：

- 使用检查点存储在步骤和系统重启之间持久化状态。
- 使用 `thread_id = session_id` 作为游戏会话和 LangGraph 线程之间的映射。

确保：

- 在会话加载时，`GameSession` 可以从现有检查点重建 LangGraph 运行器。
- 在每个新玩家消息上，引擎恢复该会话线程的图执行。

**参考文档：**

- LangGraph 文档关于：
  - 状态定义。
  - 检查点和 `MemorySaver`。
  - 工具集成。
  - 条件边和子图。

---

## 第六阶段 – 接口层（CLI 和/或最小 Web UI）

### 6.1 目标

提供一个简单的接口（首先是 CLI，稍后是可选的 Web UI）以：

- 列出可用谜题。
- 开始新的游戏会话。
- 向现有游戏会话发送消息。
- 显示 DM 响应和会话状态。

### 6.2 交付物

- CLI 命令集或小型 Web API。
- 与 `GameEngine` 和 LangGraph Agent 图的集成。

### 6.3 主要任务

#### 6.3.1 CLI 接口

设计一个最小的 CLI，能够：

- `list-puzzles` – 打印谜题 ID、标题、难度、标签。
- `start-session --puzzle <id> --player <name>` – 返回 `session_id`。
- `play --session <id>` – 交互式循环：读取玩家输入，发送到引擎，显示 DM 响应。
- `status --session <id>` – 显示当前阶段、已用回合、提示使用情况。

此 CLI 应使用：

- `GameEngine` API（列出/创建/获取会话）。
- `GameSession` + LangGraph 运行器来处理消息。

#### 6.3.2 可选 HTTP/JSON API

如果需要，设计一个适合未来 Web/聊天前端的基本 HTTP API：

- `GET /puzzles` – 列出谜题。
- `POST /sessions` – 创建会话。
- `POST /sessions/{session_id}/messages` – 发送玩家消息，获取 DM 回复。
- `GET /sessions/{session_id}` – 获取会话状态摘要。

专注于一致的 JSON 响应，包含如下字段：

- `session_id`、`puzzle_id`、`state`、`turns`、`last_message`、`last_response`。

此 API 可以使用轻量级 Web 框架实现，但此阶段不需要 UI 设计。

---

## 第七阶段 – 长期记忆增强与分析

### 7.1 目标

利用记忆基础设施提供更丰富的**玩家个性化**、**谜题分析**和**动态提示策略**。

### 7.2 交付物

- 会话结束总结流程。
- 玩家画像检索并注入 DM 提示。
- 谜题和玩家的基本分析导出。

### 7.3 主要任务

#### 7.3.1 会话结束总结

将 `summarize_session` 操作集成到 LangGraph 流程中：

- 当 `game_phase` 变为 `completed` 时：
  - 提取关键事件（提出的问题、使用的提示、最终假设、结果）。
  - 通过 LLM 生成摘要，描述：
    - 玩家推理风格。
    - 常见错误。
    - 显著优势。
  - 将摘要持久化为玩家记忆文档。
  - 可选地也将谜题统计数据（回合数、成功/失败）持久化到 `global` 记忆中。

#### 7.3.2 感知玩家画像的 DM 行为

在 DM 节点生成响应之前：

- 调用 `retrieve_player_profile(player_id)`。
- 将检索到的画像注入系统提示，以便 DM 可以：
  - 调整难度。
  - 定制解释。
  - 选择提示强度。

通过 `agents.yaml` 配置使此行为可配置（例如，DM 应多大程度上依赖画像）。

#### 7.3.3 分析与监控

设计基本分析输出：

- 聚合来自 `game_storage/events/` 的日志以计算：
  - 每个谜题的平均问题数。
  - 每个谜题的成功率。
  - 常见会话时长。

- 提供脚本或例程以：
  - 将聚合统计数据导出为 CSV/JSON 报告。
  - 可选地将高级摘要写入全局记忆文档。

引入可观测性钩子：

- 围绕关键事件（会话开始/结束、提示使用、假设裁决）的结构化日志记录。
- 如果需要，可选地跟踪 LangGraph 执行（例如，使用 LangSmith 或类似工具）。

---

## 第八阶段 – 高级功能与扩展

### 8.1 目标

一旦核心游戏循环稳定，就使用更高级的功能扩展系统。

### 8.2 潜在扩展

1. **AI Player Agent**
   - 实现一个可选的 AI 玩家，仅使用**公开** RAG 上下文和会话历史自动玩游戏。
   - 作为额外的节点/子图集成到 LangGraph 中。

2. **战役 / 多谜题模式**
   - 允许跨越多个谜题的会话（战役），并在谜题之间共享玩家记忆。
   - 扩展 `Game` 和 `GameSession` 模型以支持谜题序列。

3. **新游戏类型（谋杀之谜等）**
   - 创建额外的 `GameDataLoader` 变体或谜题模式。
   - 定义新的 RAG 元数据类型和访问规则。

4. **可插拔存储后端**
   - 将存储抽象在接口后面，允许从文件系统切换到数据库（例如 SQLite、Postgres），而无需更改游戏逻辑。

5. **管理 / 作者工具**
   - 用于摄入新谜题、验证元数据和预览 DM 将如何呈现它们的工具。

---

## 非功能性要求与质量门槛

### 性能与可扩展性

- 单会话性能应足以满足交互式使用（延迟主要由 LLM 调用决定）。
- 保持 RAG 查询按 `kb_id`（`game_<puzzle_id>`）范围划分，以最小化向量搜索成本。
- 在适当的地方使用缓存（例如，缓存谜题陈述）。

### 安全与保障

- 防止解决方案泄露：
  - 严格执行 RAG 元数据过滤器。
  - 确保 DM/Judge 的提示清楚地指示模型**不要**过早泄露完整答案。

- 缓解提示注入：
  - 将玩家输入视为不可信内容。
  - 避免在未清理的情况下将其直接连接到结构化工具控制提示中。

### 可观测性

- 使用结构化日志记录：
  - 会话创建/结束。
  - 关键状态转换。
  - 提示使用。
  - 来自 RAG 或模型提供商的错误。

- 可选地与 LangGraph 流程的跟踪工具集成。

### 测试策略

- 单元测试：
  - 领域模型、仓库和存储。
  - 记忆操作和总结逻辑（使用存根 LLM）。

- 集成测试：
  - 使用返回确定性响应的存根 LLM 进行端到端会话。
  - RAG 集成测试，验证按元数据过滤文档的正确性。

- 手动冒烟测试：
  - 使用实际模型对少量谜题进行 CLI 驱动的试玩。

---

## 现有仓库结构的阶段映射

- `src/rag/`
  - 被 `KnowledgeBaseManager` 复用和包装。

- `data/situation_puzzles/`
  - `PuzzleRepository` 和 `GameDataLoader` 的谜题来源。

- `rag_storage/`
  - 由 `KnowledgeBase` 管理的谜题知识库存储。

- 新目录：
  - `config/` – 配置文件（游戏、模型、Agent）。
  - `game_storage/` – 会话、事件和玩家记忆。
  - `src/game/`（或类似） – 领域模型、引擎、会话、记忆抽象、LangGraph 编排。

