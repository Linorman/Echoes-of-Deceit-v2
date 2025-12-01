# Echoes of Deceit v2

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/LangChain-Powered-purple.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-Powered-orange.svg" alt="LangGraph">
</p>

**Echoes of Deceit v2** is an AI-powered **Turtle Soup (Situation Puzzle) Game System**. Turtle Soup is a classic lateral thinking puzzle game where players ask yes/no questions to deduce the hidden truth behind a mysterious scenario.

## ✨ Features

- **Multi-Agent AI System**: Includes DM (Dungeon Master), Judge, Player Agent, and Hint Provider roles
- **LangChain Integration**: Unified model abstraction using LangChain for seamless provider switching
- **RAG Knowledge Base**: Powered by LightRAG/MiniRAG for puzzle knowledge management and retrieval-augmented generation
- **LangGraph Workflow**: State graph-based game flow orchestration
- **Multiple Model Support**: Supports Ollama local models and OpenAI-compatible APIs via LangChain
- **Memory Management**: Session memory, player profiles, and game analytics
- **Dual Interface**: Both CLI and Web UI (Streamlit) available
- **Internationalization**: Supports multiple languages (English & Chinese)

## 🎮 Game Rules

1. **Start**: The DM presents a mysterious situation description (the puzzle)
2. **Question**: Players explore the truth through yes/no questions
3. **Judgment**: System returns `YES` / `NO` / `YES_AND_NO` / `IRRELEVANT`
4. **Hint**: Players can request hints (with usage limits)
5. **Hypothesis**: Players propose their hypothesis when they think they understand
6. **End**: Correct hypothesis wins the game; score is calculated

## 📁 Project Structure

```
Echoes-of-Deceit-v2/
├── config/                      # YAML configuration files
│   ├── agents.yaml              # Agent role configuration
│   ├── game.yaml                # Game system configuration
│   └── models.yaml              # LLM/Embedding model configuration
├── data/                        # Puzzle data
│   └── situation_puzzles/       # Situation puzzles directory
├── rag_storage/                 # RAG knowledge base storage
├── game_storage/                # Game session data storage
├── src/                         # Source code
│   ├── cli.py                   # System management CLI
│   ├── play.py                  # Game interaction CLI
│   ├── config/                  # Configuration loading module
│   ├── game/                    # Game core module
│   ├── models/                  # LLM client module
│   ├── rag/                     # RAG knowledge base module
│   └── webui/                   # Streamlit Web UI
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── requirements.txt             # Python dependencies
└── pytest.ini                   # pytest configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (for local models) or OpenAI API key

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Linorman/Echoes-of-Deceit-v2.git
cd Echoes-of-Deceit-v2
```

2. **Create a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure models**

Edit `config/models.yaml` to set up your model provider:

```yaml
# For Ollama (local)
provider: ollama
ollama:
  base_url: http://localhost:11434
  llm_model_name: qwen3:4b-instruct-2507-q4_K_M
  embedding_model_name: qwen3-embedding:4b

# For OpenAI API
provider: api
api:
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  llm_model_name: gpt-4o-mini
  embedding_model_name: text-embedding-3-small
```

### Running the Game

#### CLI Mode

**System Management CLI** - Manage puzzles and knowledge bases:

```bash
cd src
python cli.py --help
python cli.py config          # Show current configuration
python cli.py puzzles         # List available puzzles
python cli.py kbs             # List knowledge bases
python cli.py build <puzzle>  # Build knowledge base for a puzzle
```

**Game CLI** - Play the game:

```bash
cd src
python play.py --help
python play.py list                    # List available puzzles
python play.py start <puzzle_id>       # Start a new game
python play.py start <puzzle_id> --human  # Play as human (not AI agent)
python play.py resume <session_id>     # Resume a saved session
```

#### Web UI Mode

Launch the Streamlit web interface:

```bash
cd src
streamlit run webui/app.py
```

Then open your browser at `http://localhost:8501`

## 🏗️ Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| `GameEngine` | Central controller for game lifecycle and resource management |
| `GameSessionRunner` | Manages individual game sessions and turn-based interactions |
| `KnowledgeBaseManager` | RAG knowledge base creation and querying |
| `MemoryManager` | Session and player memory persistence |
| `ModelProviderRegistry` | Unified access to LangChain-based LLM and embedding models |
| `GameGraphBuilder` | LangGraph workflow construction |

### Agent Roles

| Agent | Role |
|-------|------|
| **DM (Dungeon Master)** | Presents puzzles, controls information flow, judges player questions |
| **Judge** | Evaluates player questions and hypotheses against the truth |
| **Player Agent** | AI player that asks strategic questions to solve the puzzle |
| **Hint Provider** | Generates contextual hints when requested |

### Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Player    │────▶│    Judge    │────▶│     DM      │
│  (Question) │     │ (Evaluate)  │     │  (Response) │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       │              ┌─────────────┐          │
       └─────────────▶│  RAG Query  │◀─────────┘
                      │ (Knowledge) │
                      └─────────────┘
```

## 📝 Creating Custom Puzzles

Create a new puzzle in `data/situation_puzzles/<puzzle_name>/`:

```json
{
  "puzzle": "A man walks into a restaurant and orders albatross soup. After tasting it, he leaves and commits suicide. Why?",
  "answer": "The man was a shipwreck survivor. During the ordeal, he was told he was eating albatross soup, but it was actually the flesh of his deceased wife. When he tasted real albatross soup at the restaurant, he realized the truth.",
  "additional_info": [
    {
      "shipwreck": "The man survived a shipwreck on a deserted island",
      "wife": "His wife died during the ordeal",
      "deception": "Other survivors fed him meat claiming it was albatross"
    }
  ]
}
```

Then build the knowledge base:

```bash
python cli.py build <puzzle_name>
```

## ⚙️ Configuration

### Game Settings (`config/game.yaml`)

```yaml
rag:
  default_provider: lightrag  # or minirag

game:
  default_language: en
  max_turn_count: 100
  default_hint_limit: 5
```

### Agent Settings (`config/agents.yaml`)

Configure DM persona, judge strictness, hint strategy, and more.

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

## 📚 Documentation

- [Architecture Documentation](ARCHITECTURE.md)
- [System Design (English)](docs/system_design_turtle_soup_agents_en.md)
- [System Design (Chinese)](docs/system_design_turtle_soup_agents_zh.md)
- [Development Plan (English)](docs/development_plan_turtle_soup_agents_en.md)
- [Turtle Soup Rules](docs/turtle_soup_rules_en.md)

## 🛠️ Technology Stack

- **Python 3.10+**
- **LangChain** - Unified LLM and embedding model abstraction
- **LangGraph** - Agent orchestration and workflow management
- **LightRAG / MiniRAG** - Retrieval-augmented generation
- **Ollama** - Local LLM inference (via langchain-ollama)
- **OpenAI API** - Cloud LLM support (via langchain-openai)
- **Streamlit** - Web UI framework
- **Pydantic** - Data validation
- **PyYAML** - Configuration management

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

**Linorman**

---

<p align="center">
  <i>Uncover the truth, one question at a time. 🔍</i>
</p>
