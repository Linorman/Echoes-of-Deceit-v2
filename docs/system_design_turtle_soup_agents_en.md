# Echoes of Deceit – Turtle Soup Multi‑Agent System Design

> Draft date: 2025‑11‑30

## 1. Vision and Goals

This project implements a **multi‑agent "Turtle Soup" (situation puzzle) game system** where agents can take on different roles:

- **DM (Dungeon Master / Narrator / Referee)** – presents the puzzle, controls information flow, manages the world state, and acts as the rules arbiter for player questions and hypotheses.
- **Player Agent(s)** – ask yes/no/yes‑and‑no/irrelevant questions, hypothesize about the solution, and try to converge to the truth.

The system leverages:

- **RAG (Retrieval‑Augmented Generation)** for puzzle knowledge and world consistency (already implemented under `src/rag/`).
- **Long‑term memory** to track game progress, player behavior, and persistent knowledge across sessions.
- **LangChain and LangGraph** for agent orchestration and stateful conversational workflows.
- **Pluggable model providers** supporting both **Ollama (local)** and **remote API‑based models** (e.g., OpenAI‑compatible endpoints).

Primary goals:

1. A **modular architecture** where puzzle data, game logic, and agent roles are cleanly separated.
2. Support for **multiple games / puzzles** with isolated knowledge bases and memories.
3. **Reproducible game sessions** (save / load state) and observability (logs, traces, replay).
4. Easy extensibility for new roles, new RAG providers, and new model backends.

---

## 2. High‑Level Architecture

At a high level, the system consists of the following layers:

1. **Interface Layer**
   - Channels: CLI, web UI, chat (e.g., Discord/Slack bot in the future).
   - Responsible for user authentication (if any), session management, and mapping user messages to LangGraph entry points.

2. **Orchestration Layer (LangGraph / LangChain)**
   - Defines **agent graphs** representing game flows: DM and Player agents and their interactions.
   - Manages conversation state, routing, and tool calls.

3. **Game Logic Layer**
   - Game session manager, puzzle loader, rule engine, scoring system.
   - Integrates with the RAG layer to answer content queries.
   - Integrates with Memory layer to persist and query player/game history.

4. **Knowledge and Memory Layer**
   - **RAG subsystem** (already implemented in `src/rag/`):
     - `KnowledgeBase`, `RAGProviderFactory`, `LightRAGProvider`, `MiniRAGProvider`, `GameDataLoader`.
   - **Long‑term memory**:
     - Per‑session and per‑player memory stores.
     - Embedding + vector storage (could reuse existing RAG stores or a separate memory store).

5. **Model Provider Layer**
   - Abstractions for LLM and embedding models that can be backed by:
     - Ollama
     - OpenAI‑compatible HTTP APIs (OpenAI, Azure OpenAI, other providers).

6. **Persistence and Storage Layer**
   - File‑based storage for puzzles, knowledge bases, and RAG artifacts (already present under `data/` and `rag_storage/`).
   - Databases or files for session logs and long‑term memory.

7. **Configuration and Observability**
   - Central configuration files (e.g., YAML/JSON) for choosing RAG provider, LLM backend, and puzzle set.
   - Logging, metrics hooks, and optional tracing of LangGraph flows.

### 2.1 Component Diagram (Conceptual)

Main components and relationships:

- `GameEngine`
  - Uses `PuzzleRepository` to load puzzle metadata.
  - Interacts with `KnowledgeBaseManager` (wrapper over existing `KnowledgeBase`).
  - Spawns `GameSession` instances.

- `GameSession`
  - Holds references to the active `PuzzleContext`, `MemoryManager`, and `AgentGraph` (LangGraph graph).
  - Exposes methods `handle_player_message`, `save`, `load`, etc.

- `AgentGraph` (LangGraph)
  - Nodes: `DMNode`, `PlayerNode`, `ToolInvocationNode`, `RAGQueryNode`, `MemoryUpdateNode`.
  - Edges define the flow between nodes depending on game state.

- `MemoryManager`
  - Interfaces with **long‑term memory storage**:
    - Episode memory: game‑specific.
    - Semantic memory: player profiles, recurring patterns.

- `ModelProviderRegistry`
  - Provides unified access to `LLMClient` and `EmbeddingClient` based on configuration (`provider=ollama` or `provider=api`).

---

## 3. Game Model and Domain Concepts

### 3.1 Core Domain Entities

- **Puzzle**
  - `id`: unique identifier (e.g., `puzzle1`, `puzzle_1_en`).
  - `title`: puzzle title.
  - `description`: high‑level description.
  - `puzzle_statement`: the text shown to the players.
  - `answer`: canonical solution text.
  - `constraints`: special rules (e.g., max questions, question types).
  - `tags`: difficulty, theme, etc.

- **Game**
  - Could map 1‑to‑1 with a puzzle (simple mode) or encompass multiple puzzles in a campaign (advanced mode).

- **GameSession**
  - `session_id`: unique session identifier.
  - `puzzle_id`: associated puzzle.
  - `players`: one or more `PlayerProfile` objects.
  - `state`: `LOBBY`, `IN_PROGRESS`, `COMPLETED`, `ABORTED`.
  - `turn_history`: chronological list of turns, including questions, answers, and meta events.
  - `config`: LLM / RAG / memory configuration for this session.

- **PlayerProfile**
  - `player_id` (or `user_id`).
  - `display_name`.
  - `preferences` (difficulty, language, etc.).
  - `long_term_memory_ref`: ID/pointer to their memory store.

- **AgentRole**
  - `DM`, `Player`, `Observer` (future), `HintMaster` (future, specialized hint bot).

### 3.2 Data Sources

- **Puzzle JSON files** in `data/situation_puzzles/...` (e.g., `puzzle_1_en.json`):
  - Already ingested into RAG using `GameDataLoader` and `KnowledgeBase`.

- **RAG Storage** in `rag_storage/` and `demo/bak/01/rag_storage/`:
  - `kb_registry.json` mapping games to `game_puzzle1`, `game_puzzle2`, etc.
  - Subdirectories per game (`game_puzzle1`, `game_puzzle2`), storing vector DB, graph, and KV stores.

### 3.3 Game Flow (Simplified)

1. Player starts a new game session and selects a puzzle (or random puzzle).
2. System creates a `GameSession` with associated `KnowledgeBase` and `AgentGraph`.
3. DM introduces the puzzle statement (using RAG / structured puzzle text).
4. Player asks questions.
5. DM, acting as the referee, evaluates the question, consults the hidden answer and RAG store, then responds with:
   - `YES`, `NO`, `YES AND NO`, `IRRELEVANT`, or a limited natural language explanation, depending on game rules.
6. Game continues until the player proposes a final hypothesis.
7. DM evaluates the hypothesis against the canonical answer and returns a verdict.
8. DM reveals full explanation, optional hints, and logs completion.

---

## 4. RAG Subsystem Integration

The RAG subsystem is already implemented and used for puzzle knowledge. This design integrates it as follows.

### 4.1 Existing RAG Components (Summary)

- `KnowledgeBase` (`src/rag/knowledge_base.py`)
  - Manages multiple knowledge bases for puzzle/scenario games.
  - Uses `_REGISTRY_FILE = "kb_registry.json"` under a base storage directory.
  - Each knowledge base has a `KnowledgeBaseConfig` with fields like `kb_id`, `name`, `description`, `working_dir`, `provider_type`, `metadata`, `status`.
  - Provides async methods:
    - `create_knowledge_base`, `delete_knowledge_base`.
    - `insert_documents(kb_id, documents)`.
    - `query(kb_id, query, **kwargs)`.
    - `health_check(kb_id)`, `close_all()`.

- `RAGProviderFactory` (`src/rag/provider_factory.py`)
  - Maps provider names (`"lightrag"`, `"minirag"`) to provider classes.
  - Creates provider instances (`LightRAGProvider`, `MiniRAGProvider`) from `RAGConfig`.

- `LightRAGProvider` and `MiniRAGProvider`
  - Wrap different RAG engines (LightRAG, MiniRAG) with a common `BaseRAGProvider` interface.
  - Support async `ainitialize`, `ainsert`, `aquery`, `astream_query`, `ahealth_check`.
  - Handle configuration for LLM and embedding backends (including Ollama and OpenAI‑style APIs).

- `GameDataLoader` (`src/rag/tools/kb_loader.py`)
  - Converts puzzle JSONs into `RAGDocument`s.
  - Creates documents for puzzle statement, answer, hints, additional info, etc.

### 4.2 Mapping Games to Knowledge Bases

Each puzzle/game is mapped to an isolated RAG knowledge base:

- `kb_id` naming convention: `game_<puzzle_id>` (e.g., `game_puzzle1`).
- `kb_registry.json` stores metadata, including:
  - `game_id`, `game_type`, `source_files`, `document_count`.

At **game session creation time**, the engine will:

1. Look up or create the knowledge base for the selected puzzle:
   - On first run: use `GameDataLoader` to load puzzle JSON -> `RAGDocument`s.
   - Call `KnowledgeBase.create_knowledge_base(...)` then `insert_documents(...)`.
2. Store `kb_id` in the `GameSession`.
3. Use `KnowledgeBase.query(kb_id, query, **kwargs)` when agents need puzzle‑related information.

### 4.3 RAG Usage by Agents

- **DM Agent**
  - Uses RAG to fetch puzzle statement and non‑spoiler background details.
  - Can also use RAG to generate hints by retrieving partial answer/hint documents.

- **Player Agent (AI‑controlled)**
  - Can use RAG to recall previously seen facts or puzzle description, not hidden answers.
  - Access should be restricted so that the player agent can only see what a human player would know.

### 4.4 Access Control in RAG

To protect the solution from being leaked:

- Documents in the knowledge base are tagged via `metadata`:
  - `"type": "puzzle_statement"` – public.
  - `"type": "puzzle_answer"` – secret (Judge/DM only).
  - `"type": "hint"` – DM‑controlled.
  - `"type": "additional_info"` – DM/Judge only, may be turned into hints.

- The RAG query interface exposed to each agent will apply **metadata filters**:
  - Player’s RAG tool queries only `type in {puzzle_statement, public_fact}`.
  - Judge’s RAG tool queries all types, including `puzzle_answer`.
  - DM’s RAG tool can query hints and extra info.

This is implemented via **wrapped RAG tools** that internally call `KnowledgeBase.query(...)` with appropriate filters.

---

## 5. Memory Design with LangChain / LangGraph

The system’s memory is split into **short‑term conversational memory** and **long‑term semantic / episodic memory**.

- Within a single session, memory lets agents understand “what has happened so far” in the current game.
- Across sessions, memory lets the system remember player profiles, performance, and global statistics to enable personalization and strategic adaptation.

The design explicitly reuses **LangGraph’s long‑term memory (Store + namespaces)** and **LangChain’s short‑term conversation memory components**, instead of building everything from scratch.

### 5.1 Memory Layers and Types

Following the memory taxonomy used in the LangChain docs (semantic / episodic / procedural), the system defines three types of memory and maps them onto LangChain / LangGraph primitives:

1. **Session (Episodic) Memory – Short‑Term / Thread‑Scoped**

   - Tightly bound to a single `GameSession`, representing one playthrough of a turtle soup puzzle.
   - Primary purposes:
     - Provide DM / Player agents with context of “what has happened in this game” (questions, rulings, hints, narration, etc.).
     - Enable recaps, clarifications, and state‑dependent rulings.
   - Mapping to LangChain / LangGraph:
     - **LangGraph graph state + checkpointing**:
       - Each session corresponds to a LangGraph thread (`thread_id = session_id`).
       - The graph state includes fields like `turn_history`, `game_phase`, `hint_count`, etc.
       - LangGraph’s checkpointing (e.g., `MemorySaver` or a custom checkpoint store) persistently stores this state on every step, providing thread‑scoped short‑term memory.
     - **LangChain short‑term conversation memory (optional)**:
       - Within specific DM / Player agent nodes, when we want to pass a “recent N turns” chat history into the model, we can use:
         - `ConversationBufferWindowMemory`, or
         - a custom implementation based on `BaseChatMessageHistory`.
       - These components are populated from the graph state / event log and converted into a list of chat messages suitable for LLM prompts.

2. **Player Long‑Term Memory (Cross‑Session) – Semantic Memory**

   - Associated with `PlayerProfile`, storing aggregated summaries across many sessions, such as:
     - Preferred questioning style, common mistakes, risk tolerance.
     - Average number of questions per puzzle, success rate, abandon rate.
   - Primary purposes:
     - Inject player profiles into DM system prompts for personalized narration and difficulty tuning.
     - Drive hinting strategy (e.g., more direct hints for consistently stuck players).
   - Mapping to LangChain / LangGraph:
     - Use **LangGraph’s `Store`** as the long‑term semantic memory backend:
       - One namespace per player, e.g., `namespace = "player:<player_id>"`.
       - Store a collection of “memory documents”:
         - `id`: document key (e.g., `profile`, `performance_2025Q1`).
         - `content`: natural‑language or structured summary of the player.
         - `metadata`: `{ "summary_type": "style_profile" | "performance_summary", "last_updated": ... }`.
       - Use the Store’s built‑in **semantic search** and **content filtering** (`query` + `filter`) to fetch the most relevant player memories before a DM / Judge node runs, and inject them into its system prompt.

3. **Global Memory (System‑Level) – Semantic / Procedural Memory**

   - Not tied to any specific player or session; captures system‑wide knowledge and strategies, such as:
     - Which hint strategies generally improve user experience.
     - Global puzzle difficulty statistics and common misconceptions.
     - (Optional) system‑level “rules of thumb” that can be treated as procedural memory.
   - Mapping to LangChain / LangGraph:
     - Also backed by **LangGraph Store**, but with a global namespace, e.g., `namespace = "global"`.
     - Example documents:
       - `{ "summary_type": "hint_strategy", "content": "For new players, keep the first 3 hints deliberately vague...", "last_updated": ... }`.
       - `{ "summary_type": "puzzle_stats", "puzzle_id": "puzzle1", "content": "Average 18 questions, 35% success rate" }`.
     - Offline tools or maintenance jobs can periodically update these documents based on session logs, enabling analytics and future tuning.

### 5.2 Data Models and Storage Choices

#### 5.2.1 Session Event (Episodic Memory)

Session events exist both in the LangGraph state (for online reasoning) and optionally in the LangGraph Store (for offline analysis and cross‑session aggregation). Typical fields:

- `session_id`
- `turn_index`
- `timestamp`
- `role`: `DM` / `Judge` / `Player` / `System`
- `message`: natural‑language content (question, answer, narration, etc.)
- `tags`: e.g., `question`, `answer`, `hint`, `hypothesis`, `final_verdict`
- `raw_tool_calls`: optional, for debugging and replay

At the storage level:

- **LangGraph state**: `turn_history` is a time‑ordered array of events that is automatically persisted in checkpoints.
- **LangGraph Store**: under `session:<session_id>` namespace, events can be appended as individual documents to support later semantic search and aggregation.

#### 5.2.2 Player Memory Record (Semantic Memory)

Player memories are primarily stored as summary documents in the LangGraph Store:

- `player_id`
- `summary_type`: e.g., `style_profile`, `performance_summary`
- `content`: natural‑language summary (may embed structured data, e.g., as JSON)
- `embedding`: managed by the Store / vector backend for semantic search
- `last_updated`

When a game ends or reaches a key milestone, the system:

1. Reads important events from the `session:<session_id>` namespace.
2. Calls an LLM to generate a concise summary of the player’s behavior and performance.
3. Writes this summary as a new or updated document into the `player:<player_id>` namespace.

#### 5.2.3 Storage Backend and Relationship to RAG

- **Preferred option: LangGraph’s built‑in Store**
  - Use LangGraph’s official `Store` abstraction to manage memory documents, leveraging its native support for semantic search and filtering.
  - Benefit: tightly integrated with LangGraph’s long‑running agent and durable execution features; no need to manage a separate index.

- **Advanced option: Custom Store backend reusing existing RAG infrastructure**
  - If we want a single underlying vector / KV infrastructure for both puzzle knowledge and memory, we can implement a custom `Store` backend that:
    - Maps `Store.put` / `Store.search` to the existing `KnowledgeBase` / LightRAG / MiniRAG vector store APIs.
  - This lets puzzle knowledge (`kb_id = game_puzzle1`, etc.) and long‑term memory share the same low‑level tech stack, while remaining logically isolated via namespaces.

### 5.3 Memory Operations and Tooling

Memory operations are surfaced as **tools or dedicated LangGraph nodes** and integrated into the game flow:

1. `append_session_event(session_id, event)`

   - Role: invoked after each turn.
   - Behavior:
     - Append `event` to `turn_history` in the LangGraph state.
     - (Optional) Also write `event` as a document under `session:<session_id>` in the Store for offline analysis and long‑term episodic memory.

2. `get_session_history(session_id, limit=N)`

   - Role: provide DM / Player agents with the last N turns of conversational context.
   - Behavior:
     - Read the tail of `turn_history` from the current thread’s LangGraph state.
     - For longer history windows, fall back to reading from the `session:<session_id>` namespace in the Store.

3. `summarize_session(session_id)`

   - Role: run when a game completes (or at key checkpoints) to produce a high‑level summary.
   - Behavior:
     1. Load key events from the LangGraph state and/or the `session:<session_id>` namespace.
     2. Use an LLM chain to generate a natural‑language summary: player strategy, common mistakes, notable decisions, etc.
     3. Write this summary into `player:<player_id>` and, when relevant, into the `global` namespace (e.g., updating difficulty statistics).

4. `update_player_profile(player_id, new_info)`

   - Role: merge new information into the long‑term player profile.
   - Behavior:
     - Retrieve existing profile documents from `player:<player_id>` (by `summary_type` and/or semantic search).
     - Call an LLM to merge the old profile with `new_info` into a unified summary.
     - Overwrite the `profile` document with the updated summary.

5. `retrieve_player_profile(player_id)`

   - Role: execute before DM / Judge agents respond to a player.
   - Behavior:
     - Perform a semantic search over `player:<player_id>` to select the 1–3 most relevant profile / performance documents.
     - Inject these summaries into the system prompt of DM / Judge agents to influence tone, difficulty, and hint strategy.

Operationally, these functions are implemented as LangChain tools and LangGraph nodes. They present a simple API to the game engine while internally leveraging LangGraph’s long‑term memory and LangChain’s conversational memory features.

---

## 6. Model Provider Abstraction

The system must support using either **Ollama** or **API‑based models** with minimal changes to game logic.

### 6.1 LLM and Embedding Clients

Define abstract interfaces (pseudo‑contracts):

- `LLMClient`
  - `generate(prompt, **params) -> str` (sync or async variant).
  - `stream(prompt, **params) -> AsyncIterator[str]`.

- `EmbeddingClient`
  - `embed(texts: List[str]) -> List[List[float]]`.

Implementation classes:

1. **OllamaLLMClient / OllamaEmbeddingClient**
   - Use local Ollama HTTP endpoints.
   - Configurable base URL, model name, temperature, etc.

2. **OpenAICompatibleLLMClient / OpenAICompatibleEmbeddingClient**
   - Use OpenAI / Azure / other API‑compatible services.
   - Configurable base URL, API key, model name, extra parameters.

These clients are used by:

- Existing RAG providers (`LightRAGProvider`, `MiniRAGProvider`) through their configuration options.
- New LangChain `ChatModel` and `Embeddings` wrappers in the agent layer.

### 6.2 Provider Selection

Central configuration (YAML/JSON, e.g., `config/model_providers.json`):

- Example settings:

  - `provider: "ollama" | "api"`.
  - `llm_model_name`, `embedding_model_name`.
  - `api_key`, `base_url`.
  - `default_temperature`, `max_tokens`, etc.

At startup:

- A `ModelProviderRegistry` reads config and returns instantiated `LLMClient` and `EmbeddingClient`.
- RAG components are configured accordingly through `KnowledgeBase` / provider options.
- LangChain models are constructed from these low‑level clients.

---

## 7. Orchestration with LangChain and LangGraph

### 7.1 Overall Graph Structure

Use LangGraph to define the **game state machine** and interactions among agents.

Recommended top‑level graph:

- Entry node: `PlayerMessageNode` (receives user input from UI).
- Routing logic based on message type (question, hypothesis, command).
- Sub‑graphs:
  - **Question Flow** – Player asks a question.
  - **Hypothesis Flow** – Player proposes a solution.
  - **Hint Flow** – Player explicitly or implicitly asks for a hint.

Each flow involves specific roles:

- **DM Node** – for narrative, instructions, recap, and for making rules decisions on questions/hypotheses.
- **RAG Tool Node** – for knowledge retrieval.
- **Memory Update Node** – to persist events and update summaries.

### 7.2 Agent Definitions (Conceptual)

#### DM Agent

Responsibilities:

- Introduce the puzzle.
- Give clarifications that do not reveal the answer.
- Provide story flavor and immersion.
- Classify player inputs as **question**, **hypothesis**, or **meta command** (e.g., `restart`, `hint`).
- For questions: answer with `YES` / `NO` / `YES AND NO` / `IRRELEVANT` plus optional short remarks.
- For hypotheses: determine whether they match the puzzle answer or are partially correct, and decide whether the game should end.

Inputs:

- Puzzle metadata and full RAG context (including answer and hints).
- Session history (last N turns) and player long‑term memory.

Tools:

- `get_full_puzzle_context(kb_id)` – RAG query with all document types.
- `append_session_event`.
- `update_session_state`.

#### Player Agent (Optional AI Player)

In some modes, an AI Player agent can be used to auto‑play puzzles.

Responsibilities:

- Generate good questions.
- Refine hypotheses based on DM’s rulings and hints.

Inputs:

- Public puzzle statement and conversation history.
- (Optional) limited RAG access to public documents.

Tools:

- `get_public_puzzle_context(kb_id)`.
- `append_session_event`.

### 7.3 State and Memory in LangGraph

Define a **graph state object** (for LangGraph) with fields like:

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

State transitions are controlled by nodes and conditional edges. For example:

1. `PlayerMessageNode` updates `last_user_message` and calls a **classifier** (LLM) to set `message_type`.
2. If `message_type == "question"` -> send to `DMNode`.
3. If `message_type == "hypothesis"` -> send to `DMNode`.
4. After each node, call `MemoryUpdateNode` to persist events.
5. If `game_phase` becomes `completed`, route to `RevealSolutionNode`.

### 7.4 Tools in LangChain

Expose key operations as tools:

- `rag_query_public(kb_id, query, **filters)`.
- `rag_query_full(kb_id, query, **filters)`.
- `append_event(session_id, event)`.
- `get_recent_events(session_id, limit)`.
- `summarize_session(session_id)`.
- `get_player_profile(player_id)`.
- `update_player_profile(player_id, summary)`.

These tools are used by agents defined as `Tool‑using LLM Chains` or `Agents` in LangChain, then embedded into LangGraph nodes.

---

## 8. Game Engine and Session Management

### 8.1 GameEngine Responsibilities

- Initialization:
  - Load configuration (model provider, RAG provider, base directories).
  - Initialize `KnowledgeBase` with `base_storage_dir` (e.g., `rag_storage/`).
  - Initialize `ModelProviderRegistry` and LangChain models.

- Game lifecycle:
  - Discover available puzzles via `GameDataLoader.discover_games(...)`.
  - Create new sessions and associated `kb_id`.
  - Manage session lookup (by `session_id`) and persistence.

- API surface (conceptual):
  - `list_puzzles() -> List[PuzzleSummary]`.
  - `create_session(puzzle_id, player_id, **options) -> GameSession`.
  - `handle_message(session_id, player_message) -> system_response`.
  - `get_session_state(session_id) -> SessionState`.

### 8.2 GameSession Responsibilities

- Holds runtime state and references to:
  - `AgentGraph` instance.
  - `KnowledgeBase` / `kb_id`.
  - `MemoryManager`.
  - `PlayerProfile`.

- Orchestrates calls to LangGraph with a persistent state object.

- Provides methods:
  - `process_player_input(message: str) -> str | RichResponse`.
  - `save()` – persists state and memory snapshots.
  - `load(...)` – reconstructs from storage.

### 8.3 Error Handling and Health Checks

- Before launching sessions, the engine can call `KnowledgeBase.health_check(kb_id)` to ensure RAG providers are available.
- If a provider is unavailable, the system can:
  - Fallback to a simpler mode (no RAG, only static puzzle text).
  - Show a friendly error to the user.

---

## 9. Storage and Persistence

### 9.1 Existing RAG Storage

Already structured as:

- `rag_storage/kb_registry.json` – registry of knowledge bases.
- `rag_storage/game_puzzleX/` – per‑KB storage with:
  - `graph_chunk_entity_relation.graphml`
  - `kv_store_*.json`
  - `vdb_*.json`

The new design continues to use this structure and delegates all RAG‑related persistence to the existing `KnowledgeBase` and provider implementations.

### 9.2 New Storage for Sessions and Memories

Proposed directories:

- `game_storage/sessions/` – serialized session state per `session_id`.
  - e.g., `sessions/<session_id>.json` containing high‑level state (phase, score, active puzzle, etc.).

- `game_storage/events/` – append‑only logs.
  - e.g., `events/<session_id>.jsonl` with one JSON object per event.

- `game_storage/player_memory/` – player long‑term memory.
  - e.g., `player_memory/<player_id>.json` with aggregated summaries.

Optionally, use the same vector DB infrastructure as RAG for semantic memory, with a dedicated `kb_id` prefix (e.g., `memory_player_<player_id>`).

---

## 10. Configuration and Extensibility

### 10.1 Configuration Files

Create a dedicated config directory, e.g., `config/` containing:

- `config/game.yaml`
  - Default RAG provider (`lightrag` / `minirag`).
  - Base directories for data and storage.
  - Default models and parameters.

- `config/models.yaml`
  - Provider selection (`ollama` or `api`).
  - Model names, endpoints, API keys (kept out of VCS if sensitive).

- `config/agents.yaml`
  - Prompts and behavioral settings per role:
    - DM persona (tone, style).
    - Judge strictness level, allowed verbosity.
    - Hint policy (when to give hints, how strong they may be).

### 10.2 Extending the System

- **New RAG Provider**
  - Implement `BaseRAGProvider` subclass.
  - Register it via `RAGProviderFactory.register_provider(name, cls)`.

- **New Model Backend**
  - Implement `LLMClient` / `EmbeddingClient` for the new backend.
  - Extend `ModelProviderRegistry` to recognize and instantiate it.

- **New Agent Role**
  - Define the role’s prompt and tools.
  - Add LangGraph nodes and edges to integrate it into flows.

- **New Game Type (e.g., murder mystery)**
  - Implement a new `GameDataLoader` variant or extend existing one.
  - Define new document metadata types and RAG access rules.
  - Adjust game flow graph to reflect the new type.

---

## 11. Security and Safety Considerations

- **Information Leakage Prevention**
  - Strict separation of public vs secret documents via RAG filters.
  - Careful system prompts for Judge/DM instructing them not to reveal the full answer except at the end.

- **Prompt Injection Mitigation**
  - Clean inputs from players and treat them as untrusted.
  - Avoid directly concatenating player messages into tool‑control prompts.

- **Resource Limits**
  - Configurable maximum number of turns and hints per session.
  - Per‑session token budget and rate‑limiting.

---

## 12. Example End‑to‑End Flow (Narrative)

1. **Startup**
   - Load configs and initialize model providers.
   - Initialize `KnowledgeBase` with base directory `rag_storage/`.

2. **Puzzle Preparation**
   - Discover puzzles under `data/situation_puzzles/` using `GameDataLoader.discover_games`.
   - For each selected puzzle:
     - Ensure a corresponding `kb_id` exists (create + ingest if missing).

3. **Create Session**
   - Player selects `puzzle1`.
   - Engine creates `GameSession` with:
     - `kb_id = game_puzzle1`.
     - A new `session_id`.
     - A LangGraph state object with `phase = intro`.

4. **Intro Phase**
   - DM Agent fetches puzzle statement using `rag_query_public` and narrates it.
   - Session history and memory are updated.

5. **Question Phase**
   - Player submits a question.
   - Graph routes message to DM Agent.
   - DM Agent uses `rag_query_full` to inspect the answer and relevant facts, then returns `YES` / `NO` / `YES AND NO` / `IRRELEVANT` plus optional explanation.
   - DM Agent may wrap this into more narrative text for the player.
   - Memory and state (turn counter, hint usage) are updated.

6. **Hypothesis Phase**
   - Player states a full hypothesis.
   - DM Agent compares it with the canonical answer via RAG.
   - If sufficiently correct, `phase` transitions to `completed`.

7. **Resolution Phase**
   - DM Agent reveals the full answer and missing details.
   - System summarizes the session and updates `PlayerProfile` long‑term memory.

---

## 13. Implementation Roadmap (High‑Level)

1. **Foundations**
   - Introduce config files (`config/`), `ModelProviderRegistry`, and memory storage folder structure.
   - Wrap existing `KnowledgeBase` and `GameDataLoader` in a `KnowledgeBaseManager` service for game usage.

2. **Game Engine and Session Core**
   - Implement `GameEngine`, `GameSession`, `PuzzleRepository`, `MemoryManager` (structural code only at first).
   - Wire engine to the RAG subsystem and model providers.

3. **LangGraph Agent Orchestration**
   - Define graph state schema.
   - Implement basic DM nodes and a simple question flow.
   - Integrate memory update tools.

4. **Interface Layer**
   - CLI or minimal web UI for starting games and interacting with sessions.

5. **Long‑Term Memory Enhancements**
   - Add session summarization and player profile updates.

6. **Advanced Features**
   - AI Player agents.
   - Campaign mode and multiple simultaneous sessions.
   - Analytics dashboards from game logs.

This design keeps the existing RAG implementation as the foundation for puzzle knowledge while adding a structured multi‑agent, multi‑provider, memory‑aware game system on top using LangChain and LangGraph.
