# Echoes of Deceit – Multi‑Agent Turtle Soup System

## 0. Scope and Constraints

This development plan translates the Chinese system design document into a concrete, **phased implementation roadmap** for a multi‑agent "Turtle Soup" (situation puzzle) game system. It is intended for a Python backend team familiar with **LangChain**, **LangGraph**, and **RAG** patterns.

**Important constraints:**

- The plan must **fully align with the provided system design** (architecture, data flows, and responsibilities).
- The plan **MUST NOT include concrete code** (no function bodies, no complete code listings). It should focus on modules, responsibilities, APIs, and configuration.
- The plan assumes:
  - Python 3.12+.
  - Existing RAG subsystem in `src/rag/` as described (`KnowledgeBase`, providers, `GameDataLoader`).
  - Future use of LangChain & LangGraph for orchestration and memory.

The plan is divided into **phases**. Each phase can be delivered incrementally and should be shippable with basic tests and manual verification.

---

## Phase 1 – Foundations, Configuration, and RAG Integration Hardening

### 1.1 Goal

Establish a solid foundation: configuration management, model provider abstraction, and a thin **KnowledgeBaseManager** on top of the existing RAG subsystem. No LangGraph yet; focus is on **clean access to puzzle knowledge**.

### 1.2 Deliverables

- Central configuration files under `config/`.
- Model provider abstraction (`LLMClient`, `EmbeddingClient`, `ModelProviderRegistry`).
- `KnowledgeBaseManager` service wrapping `KnowledgeBase`.
- Initial CLI or simple script to:
  - Discover puzzles under `data/situation_puzzles/`.
  - Ensure corresponding KBs exist under `rag_storage/`.
  - Run basic health checks.

### 1.3 Main Tasks

#### 1.3.1 Configuration Layout

Create a `config/` directory with at least:

- `config/game.yaml`
  - Default RAG provider (e.g., `lightrag` or `minirag`).
  - Base directories:
    - `data_base_dir` -> `data/situation_puzzles/`.
    - `rag_storage_dir` -> `rag_storage/` (and possibly `demo/bak/` for compatibility).
    - Future `game_storage_dir` -> `game_storage/`.
  - Default language(s), default puzzle set, and game‑level parameters (e.g., max turn count, default hint limits).

- `config/models.yaml`
  - `provider: "ollama" | "api"`.
  - `llm_model_name` and `embedding_model_name`.
  - `api_base_url`, `api_key` or environment variable references (do not store real keys in VCS).
  - Default generation parameters (temperature, max tokens, etc.).

- `config/agents.yaml`
  - DM persona (tone, style, strictness about not revealing the answer early).
  - Judge constraints (how terse/verbose to be, how to phrase YES/NO responses).
  - Hint strategy: rules for when and how hints may be offered.

**Implementation notes (no code):**

- Use a YAML parsing library to load these configs at startup into in‑memory config objects.
- Define clear data structures representing these configs so other layers (engine, agents, RAG providers) can consume them.

#### 1.3.2 Model Provider Abstractions

Introduce a **model provider layer** that hides the details of Ollama vs API providers.

**Interfaces (conceptual):**

- `LLMClient`
  - `generate(prompt, **params) -> str` (sync or async variant depending on choice).
  - `stream(prompt, **params) -> iterator/async iterator of str` for streaming outputs.

- `EmbeddingClient`
  - `embed(texts: List[str]) -> List[List[float]]`.

**Concrete implementations:**

- `OllamaLLMClient` / `OllamaEmbeddingClient`
  - Use Ollama HTTP endpoints and configuration from `config/models.yaml`.

- `OpenAICompatibleLLMClient` / `OpenAICompatibleEmbeddingClient`
  - Use OpenAI, Azure OpenAI, or compatible APIs based on configuration.

**ModelProviderRegistry:**

- Read `config/models.yaml`.
- Based on `provider` field, instantiate the appropriate clients.
- Provide a single entry point to the rest of the system:
  - `get_llm_client()`.
  - `get_embedding_client()`.

**Docs to reference:**

- LangChain model docs (e.g., `ChatOpenAI`, `ChatOllama`), focusing on how they wrap base LLM clients.
- Provider‑specific HTTP API docs for concrete request formats.

#### 1.3.3 `KnowledgeBaseManager` Service

Wrap the existing `KnowledgeBase` class so that game code does not directly manipulate provider details.

Responsibilities:

- Initialize `KnowledgeBase` with `rag_storage_dir` and provider settings from config.
- Expose domain‑centric methods:
  - `ensure_puzzle_kb(puzzle_id) -> kb_id`.
  - `query_public(kb_id, query, filters)` (e.g., only `puzzle_statement` and public facts).
  - `query_full(kb_id, query, filters)` (including `puzzle_answer`, hints, etc.).
  - `health_check(kb_id)`.
- Use `GameDataLoader` to:
  - Discover games/puzzles in `data/situation_puzzles/`.
  - Convert puzzle JSONs into `RAGDocument` instances.
  - Insert them into the appropriate KB using `KnowledgeBase.insert_documents`.

RAG **access control** must match design:

- Use `metadata.type` flags in documents:
  - `puzzle_statement` (public),
  - `puzzle_answer` (secret),
  - `hint` (controlled),
  - `additional_info` (DM/Judge‑only).
- Implement filtering logic such that:
  - Player agents query only public types.
  - DM/Judge agents can query full context.

#### 1.3.4 CLI / Utility Script

Create a minimal CLI or main script (no complex I/O yet) that can:

- Load configs.
- Initialize `ModelProviderRegistry` and `KnowledgeBaseManager`.
- Discover puzzles and ensure KBs exist:
  - For each puzzle, create or update KB `game_<puzzle_id>`.
- Run `health_check` on each KB and print a status report.

No LangGraph at this phase. The purpose is to validate:

- Model configuration works.
- RAG pipelines and metadata tagging are correct.

---

## Phase 2 – Game Domain Model and Storage Layer

### 2.1 Goal

Define and persist the **game domain entities** (`Puzzle`, `Game`, `GameSession`, `PlayerProfile`, etc.) and basic file‑based storage. Still minimal or no LangGraph; prepare a solid domain model for later orchestration.

### 2.2 Deliverables

- Python domain model classes (or dataclasses) for all core entities.
- File‑based storage layout under `game_storage/`.
- Helper APIs for creating, loading, and updating sessions and player profiles.

### 2.3 Main Tasks

#### 2.3.1 Domain Entities

Define domain entities as Python classes (without business logic beyond simple validation and helpers):

- `Puzzle`
  - `id`, `title`, `description`.
  - `puzzle_statement`, `answer`.
  - `constraints` (e.g., max questions, language rules).
  - `tags`.

- `Game`
  - Mapping to one or multiple `Puzzle` objects (simple version: 1:1 with puzzle).

- `GameSession`
  - `session_id` (UUID or similar).
  - `puzzle_id` (string).
  - `player_ids` or `PlayerProfile` references.
  - `state`: `LOBBY`, `IN_PROGRESS`, `COMPLETED`, `ABORTED`.
  - `turn_history` (simple structured events; see memory section later).
  - `config` (LLM, RAG, memory settings snapshot taken at session creation).

- `PlayerProfile`
  - `player_id`.
  - `display_name`.
  - `preferences` (difficulty, language, etc.).
  - `long_term_memory_ref` (pointer to memory namespace or file path).

- `AgentRole`
  - Enum or similar: `DM`, `Player`, `Observer`, `HintMaster` (future).

Ensure that these model definitions are independent of LangChain/LangGraph types, focusing purely on domain.

#### 2.3.2 Puzzle Repository

Create a `PuzzleRepository` service that:

- Reads puzzle JSON from `data/situation_puzzles/`.
- Maps raw JSON fields to `Puzzle` objects.
- Provides APIs:
  - `list_puzzles() -> List[PuzzleSummary]` (id, title, difficulty, tags).
  - `get_puzzle(puzzle_id) -> Puzzle`.
  - `find_random_puzzle(filters) -> Puzzle`.

Internally, it should reuse `GameDataLoader`’s discovery logic where possible, but keep the repository interface independent of RAG details.

#### 2.3.3 Game Storage Layout

Following the design, create a directory layout under `game_storage/`:

- `game_storage/sessions/`
  - `sessions/<session_id>.json` for serialized `GameSession` state.

- `game_storage/events/`
  - `events/<session_id>.jsonl` for append‑only event logs (each line a JSON event).

- `game_storage/player_memory/`
  - `player_memory/<player_id>.json` for aggregated player summaries / profiles.

Define **serialization and deserialization rules**:

- How entities map to JSON (no circular references; use IDs for relationships).
- How enums / states are stored (e.g., strings `"IN_PROGRESS"`).
- Backwards‑compatible versioning fields for potential schema evolution (e.g., `schema_version`).

#### 2.3.4 Session Management APIs

Define a `GameSessionStore` or similar manager with methods such as:

- `create_session(puzzle_id, player_ids, options) -> GameSession`.
- `save_session(session: GameSession)`.
- `load_session(session_id) -> GameSession`.
- `list_sessions(filters) -> List[GameSession]`.

This store should be file‑based in this phase, later possibly pluggable for DB backends.

#### 2.3.5 Basic Tests and Manual Flows

- Minimal unit tests for:
  - `PuzzleRepository` (loading, listing, getting puzzles).
  - `GameSession` serialization/deserialization.
  - `GameSessionStore` read/write.

- Manual script that:
  - Creates a dummy `PlayerProfile`.
  - Creates a new session for a selected puzzle.
  - Saves and reloads the session.

---

## Phase 3 – Memory Model and Integration with LangGraph Store

### 3.1 Goal

Introduce the memory layers described in the design:

1. **Session (episodic) memory** per game session.
2. **Player‑level long‑term semantic memory**.
3. **Global system‑level memory**.

Integrate them conceptually with **LangGraph’s `Store`** abstraction while keeping a fallback file‑based representation.

### 3.2 Deliverables

- Memory model definitions for session events and player memory records.
- Abstract `MemoryStore` interface (aligned with LangGraph’s `Store` where possible).
- Concrete implementation using the file layout and optional integration with the existing RAG vector store for semantic search.

### 3.3 Main Tasks

#### 3.3.1 Session Event Schema (Episodic Memory)

Define a standard event record structure, e.g.:

- `session_id`.
- `turn_index`.
- `timestamp`.
- `role`: `DM` | `Player` | `Judge` | `System`.
- `message` (text content of question/answer/narration).
- `tags`: list of tokens such as `question`, `answer`, `hint`, `hypothesis`, `final_verdict`.
- `raw_tool_calls` (optional structured record of RAG / memory tools used that turn).

Mapping:

- Stored in LangGraph state (`turn_history` field) for online use.
- Serialized as `.jsonl` in `game_storage/events/<session_id>.jsonl` for offline analytics.

#### 3.3.2 Player Memory Records (Semantic Memory)

Define player memory summaries as documents that can later be stored in LangGraph `Store` or RAG backend:

- `player_id`.
- `summary_type`: e.g., `style_profile`, `performance_summary`.
- `content`: natural language text (possibly containing embedded JSON for structured stats).
- `metadata`: e.g., `{"summary_type": ..., "last_updated": ...}`.

Design how these summaries are created and updated:

- At end of session (or key milestones), a summarization process is triggered.
- It reads session events and produces a compact, human‑readable summary.
- That summary is merged with existing profile info (if any) and persisted.

#### 3.3.3 Global Memory Records

Define documents for global system knowledge:

- `summary_type`: e.g., `hint_strategy`, `puzzle_stats`.
- `puzzle_id` (optional, for puzzle‑specific stats).
- `content`: natural language summarizing what works well across all players and sessions.

These documents live in a global namespace (`global`) in LangGraph `Store` or in a dedicated `game_storage/global_memory/` directory.

#### 3.3.4 MemoryStore Abstraction

Define a small abstraction representing a LangGraph‑style store but decoupled from the exact implementation:

- Capabilities (conceptual):
  - `put(namespace, id, document, metadata)`.
  - `get(namespace, id)`.
  - `search(namespace, query, filters, k)`.

Implementation strategies:

- **Phase 3 baseline:**
  - File‑based documents stored under `game_storage/player_memory/` and `game_storage/global_memory/`.
  - Simple search (e.g., by metadata filter and/or naive string matching) to keep implementation light.

- **Future upgrade:**
  - Implement a `Store` backend that reuses the existing RAG vector infrastructure (LightRAG/MiniRAG) for embeddings and semantic search.

#### 3.3.5 Memory‑Related Operations (Logical APIs)

Design these operations as **logical services** (later exposed as LangChain tools / LangGraph nodes):

1. `append_session_event(session_id, event)`
   - Add event to in‑memory `turn_history`.
   - Append to `events/<session_id>.jsonl`.

2. `get_session_history(session_id, limit)`
   - Return latest `N` events from in‑memory state (or from file if needed).

3. `summarize_session(session_id)`
   - Read events.
   - Use an LLM to create a textual summary of the session and player behaviors.
   - Save summary into `player:<player_id>` namespace via `MemoryStore`.
   - Optionally update a global summary (`global` namespace).

4. `update_player_profile(player_id, new_summary)`
   - Retrieve existing profile doc for the player.
   - Merge with `new_summary` via LLM or heuristic rules.
   - Write back the updated profile document.

5. `retrieve_player_profile(player_id)`
   - Search `player:<player_id>` namespace.
   - Return 1–3 most relevant memory docs for injection into DM/Judge prompts.

At this phase, these are pure services; LangGraph integration will happen in the next phase.

**Docs to reference:**

- LangGraph `Store` documentation.
- LangChain memory types (ConversationBuffer, semantic memory patterns) for conceptual alignment.

---

## Phase 4 – Game Engine Core (Without Full LangGraph)

### 4.1 Goal

Construct the **GameEngine** and core `GameSession` runtime logic that uses the domain model, storage, and RAG subsystem, but **before** introducing the full LangGraph state machine. This allows early end‑to‑end flows in a simpler structure (e.g., synchronous function calls) and provides a fallback if LangGraph is not available.

### 4.2 Deliverables

- `GameEngine` service coordinating configs, RAG, puzzles, and sessions.
- `GameSession` runtime wrapper providing high‑level methods (`process_player_input`, etc.).
- Basic CLI loop to manually play a session through the terminal.

### 4.3 Main Tasks

#### 4.3.1 GameEngine Responsibilities

Implement a high‑level `GameEngine` component with responsibilities aligned to the design:

- Initialization:
  - Load configs (`game.yaml`, `models.yaml`, `agents.yaml`).
  - Initialize `ModelProviderRegistry`.
  - Initialize `KnowledgeBaseManager`.
  - Initialize `PuzzleRepository`.
  - Initialize storage (`GameSessionStore`, `MemoryStore`).

- Game lifecycle management:
  - `list_puzzles() -> List[PuzzleSummary]`.
  - `create_session(puzzle_id, player_id, options) -> GameSession`.
  - `get_session(session_id) -> GameSession`.
  - `save_session(session_id)` and `load_session(session_id)` wrappers.

- Health checking:
  - For a given puzzle/session, call `KnowledgeBaseManager.health_check(kb_id)`.
  - Provide human‑readable errors and fallback strategies if RAG is unavailable (e.g., degrade to static puzzle text mode).

#### 4.3.2 GameSession Runtime Logic (Pre‑LangGraph)

`GameSession` in this phase should:

- Hold references to:
  - `Puzzle`.
  - `kb_id` (from `KnowledgeBaseManager`).
  - `PlayerProfile`.
  - `turn_history` (list of session events).
  - Current `state` and phase (intro, playing, awaiting_final, completed).

- Expose a single method for the interface layer:
  - `process_player_input(message: str) -> response object`.

Internally this method should:

1. Classify input type:
   - Use a simple classification heuristic or LLM to determine if input is:
     - A **question** about the puzzle.
     - A **final hypothesis** (attempt to solve).
     - A **meta command** (e.g., restart, ask for a hint).

2. Route to appropriate internal handler:
   - Question -> DM/Judge logic (in this phase, may be a simplified rule‑based or a single LLM prompt referencing full puzzle context).
   - Hypothesis -> evaluation logic comparing hypothesis to answer.
   - Command -> handle restart, hint, or status queries.

3. Update `turn_history` and `GameSession` state appropriately.

This phase can use one or more LLM prompts, but does not yet require full LangGraph node/edge definitions.

#### 4.3.3 Simplified DM/Judge Logic

Design DM/Judge behaviors that roughly match the final desired behavior but may be implemented as **single chained prompts per step**:

- Use `KnowledgeBaseManager.query_full` to retrieve relevant info (including the hidden answer) when evaluating questions or hypotheses.
- Enforce:
  - For questions:
    - Output `YES` / `NO` / `YES AND NO` / `IRRELEVANT` plus optional brief explanation.
  - For hypotheses:
    - Output verdict (correct/incorrect/partially correct) and optional explanation.
- Implement configurable verbosity and style using `agents.yaml` configuration (DM persona).

This gives a working game loop that can be tested via CLI.

---

## Phase 5 – LangGraph‑Based Orchestration and Agent Graph

### 5.1 Goal

Refactor the runtime to use **LangGraph** for stateful, multi‑agent orchestration, matching the design’s conceptual graph: DM, Player, RAG, and memory nodes with explicit state transitions.

### 5.2 Deliverables

- LangGraph `State` schema representing `GameSession` runtime state.
- Agent graph (nodes and edges) implementing game flows:
  - Question handling.
  - Hypothesis handling.
  - Hint handling.
- Integration of memory operations and RAG as tools/nodes.

### 5.3 Main Tasks

#### 5.3.1 State Schema for LangGraph

Define a LangGraph **state object** to mirror or encapsulate `GameSession` runtime state:

- Fields (conceptual):
  - `session_id`.
  - `kb_id`.
  - `player_id`.
  - `turn_index`.
  - `last_user_message`.
  - `message_type` (`question`, `hypothesis`, `command`).
  - `last_dm_response`.
  - `game_phase` (`intro`, `playing`, `awaiting_final`, `completed`).
  - `hint_count`.
  - `score`.
  - `turn_history` (or a reference to it).

Ensure this state object is compatible with LangGraph’s checkpointing (`MemorySaver` or custom checkpoint store) so that each step is automatically persisted.

#### 5.3.2 Node Design

Define conceptual nodes in the LangGraph graph:

- `PlayerMessageNode`
  - Entry point from UI.
  - Reads raw input and populates `last_user_message`.
  - Triggers classification of `message_type` (by rule or LLM call).

- `RouteByMessageTypeNode`
  - Conditional router based on `message_type`.
  - Sends state to `QuestionFlow`, `HypothesisFlow`, or `CommandFlow` subgraphs.

- `DMQuestionNode`
  - Implements DM handling of player questions.
  - Calls RAG via `rag_query_full` tool.
  - Produces structured DM response: verdict and explanation.

- `DMHypothesisNode`
  - Compares hypothesis to answer using RAG (including `puzzle_answer`).
  - Updates `game_phase` to `completed` if solution is correct enough.

- `CommandHandlerNode`
  - Handles meta commands (restart, hint request, status query).

- `HintNode`
  - Controlled access to hint documents via RAG.
  - Uses `agents.yaml` to decide how strong/explicit hints should be.

- `MemoryUpdateNode`
  - After each DM/Player interaction, appends a session event.
  - Optionally triggers summarization once game is completed.

Each node should:

- Operate on and update the shared state object.
- Use tools for RAG, memory, and session storage rather than handling them directly.

#### 5.3.3 Tools and Integrations

Expose domain services as LangChain **Tools** usable inside LangGraph nodes:

- RAG tools:
  - `rag_query_public(kb_id, query, filters)`.
  - `rag_query_full(kb_id, query, filters)`.

- Memory tools:
  - `append_event(session_id, event)`.
  - `get_recent_events(session_id, limit)`.
  - `summarize_session(session_id)`.
  - `get_player_profile(player_id)`.
  - `update_player_profile(player_id, summary)`.

Mapping to LangGraph:

- Each tool becomes either:
  - A callable wrapped via LangChain’s tool interfaces and embedded into LLM prompts.
  - A standalone LangGraph node that performs its effect without LLM inference.

#### 5.3.4 Checkpointing and Persistence

Configure LangGraph to:

- Use a checkpoint store to persist state across steps and system restarts.
- Use `thread_id = session_id` as the mapping between game sessions and LangGraph threads.

Ensure that:

- On session load, `GameSession` can reconstruct the LangGraph runner from existing checkpoints.
- On each new player message, the engine resumes the graph execution for that session’s thread.

**Docs to reference:**

- LangGraph documentation on:
  - State definitions.
  - Checkpointing and `MemorySaver`.
  - Tools integration.
  - Conditional edges and subgraphs.

---

## Phase 6 – Interface Layer (CLI and/or Minimal Web UI)

### 6.1 Goal

Provide a simple interface (CLI first, optional web UI later) to:

- List available puzzles.
- Start a new game session.
- Send messages to an existing game session.
- Display DM responses and session status.

### 6.2 Deliverables

- CLI command set or small web API.
- Integration with `GameEngine` and LangGraph agent graph.

### 6.3 Main Tasks

#### 6.3.1 CLI Interface

Design a minimal CLI that can:

- `list-puzzles` – prints puzzle IDs, titles, difficulty, tags.
- `start-session --puzzle <id> --player <name>` – returns `session_id`.
- `play --session <id>` – interactive loop: read player input, send to engine, show DM response.
- `status --session <id>` – shows current phase, turns taken, hint usage.

This CLI should use:

- `GameEngine` APIs (list/create/get session).
- `GameSession` + LangGraph runner to process messages.

#### 6.3.2 Optional HTTP/JSON API

If desired, design a basic HTTP API suitable for a future web/Chat frontend:

- `GET /puzzles` – list puzzles.
- `POST /sessions` – create session.
- `POST /sessions/{session_id}/messages` – send player message, get DM reply.
- `GET /sessions/{session_id}` – get session state summary.

Focus on consistent JSON responses with fields such as:

- `session_id`, `puzzle_id`, `state`, `turns`, `last_message`, `last_response`.

This API can be implemented with a lightweight web framework, but no UI design is required at this phase.

---

## Phase 7 – Long‑Term Memory Enhancement and Analytics

### 7.1 Goal

Capitalize on the memory infrastructure to provide richer **player personalization**, **puzzle analytics**, and **dynamic hint strategies**.

### 7.2 Deliverables

- End‑of‑session summarization flows.
- Player profile retrieval and injection into DM prompts.
- Basic analytics exports for puzzles and players.

### 7.3 Main Tasks

#### 7.3.1 End‑of‑Session Summaries

Integrate the `summarize_session` operation into the LangGraph flows:

- When `game_phase` becomes `completed`:
  - Extract key events (questions asked, hints used, final hypothesis, outcome).
  - Generate a summary via LLM describing:
    - Player reasoning style.
    - Common mistakes.
    - Notable strengths.
  - Persist the summary as a player memory document.
  - Optionally also persist puzzle statistics (turns count, success/failure) into `global` memory.

#### 7.3.2 Player Profile‑Aware DM Behavior

Before DM nodes generate responses:

- Call `retrieve_player_profile(player_id)`.
- Inject the retrieved profile(s) into the system prompt so DM can:
  - Adjust difficulty.
  - Tailor explanations.
  - Choose hint strength.

Make this behavior configurable via `agents.yaml` (e.g., how heavily DM should rely on profiles).

#### 7.3.3 Analytics and Monitoring

Design basic analytics outputs:

- Aggregate logs from `game_storage/events/` to compute:
  - Average number of questions per puzzle.
  - Success rate per puzzle.
  - Common session lengths.

- Provide a script or routine to:
  - Export aggregated stats to a CSV/JSON report.
  - Optionally write high‑level summaries into global memory docs.

Introduce observability hooks:

- Structured logging around key events (session start/end, hint use, hypothesis verdicts).
- Optional tracing of LangGraph executions (e.g., using LangSmith or similar) if desired.

---

## Phase 8 – Advanced Features and Extensions

### 8.1 Goal

Extend the system with more advanced capabilities once the core game loop is stable.

### 8.2 Potential Extensions

1. **AI Player Agent**
   - Implement an optional AI player that uses only **public** RAG context and session history to play the game automatically.
   - Integrate as an additional node/subgraph in LangGraph.

2. **Campaign / Multi‑Puzzle Mode**
   - Allow sessions that span multiple puzzles (campaigns), with shared player memory across puzzles.
   - Extend `Game` and `GameSession` models to support puzzle sequences.

3. **New Game Types (Murder Mystery, etc.)**
   - Create additional `GameDataLoader` variants or puzzle schemas.
   - Define new RAG metadata types and access rules.

4. **Pluggable Storage Backends**
   - Abstract storage behind interfaces and allow switching from filesystem to databases (e.g., SQLite, Postgres) without changing game logic.

5. **Admin / Author Tools**
   - Tools to ingest new puzzles, validate metadata, and preview how DM would present them.

---

## Non‑Functional Requirements and Quality Gates

### Performance and Scalability

- Single‑session performance should be adequate for interactive use (latency dominated by LLM calls).
- Keep RAG queries scoped by `kb_id` (`game_<puzzle_id>`) to minimize vector search cost.
- Use caching where appropriate (e.g., caching puzzle statements).

### Security and Safety

- Prevent solution leakage:
  - Enforce RAG metadata filters rigorously.
  - Ensure prompts for DM/Judge clearly instruct the model **not** to reveal the full answer prematurely.

- Mitigate prompt injection:
  - Treat player inputs as untrusted content.
  - Avoid passing them directly into structured tool control prompts without sanitization.

### Observability

- Use structured logs for:
  - Session creation/end.
  - Key state transitions.
  - Hint usage.
  - Errors from RAG or model providers.

- Optionally integrate with tracing tools for LangGraph flows.

### Testing Strategy

- Unit tests:
  - Domain models, repositories, and storage.
  - Memory operations and summarization logic (using stub LLMs).

- Integration tests:
  - End‑to‑end session with a stub LLM that returns deterministic responses.
  - RAG integration tests verifying correct document filtering by metadata.

- Manual smoke tests:
  - CLI‑driven playthrough for a small number of puzzles using actual models.

---

## Phase Mapping to Existing Repo Structure

- `src/rag/`
  - Reused and wrapped by `KnowledgeBaseManager`.

- `data/situation_puzzles/`
  - Source of puzzles for `PuzzleRepository` and `GameDataLoader`.

- `rag_storage/`
  - Storage for puzzle KBs managed by `KnowledgeBase`.

- New directories:
  - `config/` – configuration files (game, models, agents).
  - `game_storage/` – sessions, events, and player memory.
  - `src/game/` (or similar) – domain models, engine, sessions, memory abstractions, LangGraph orchestration.
