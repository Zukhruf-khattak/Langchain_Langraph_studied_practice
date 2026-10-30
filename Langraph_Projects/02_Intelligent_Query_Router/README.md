# 02. Intelligent Query Router

An intelligent multi-agent query routing system built with **LangGraph** and **LangChain**.

## 📌 Overview

The **Intelligent Query Router** dynamically evaluates user queries and routes them to the appropriate domain-specific handler, specialized agent, or external tool based on intent, complexity, and domain.

### 🔄 Architecture Flow

```
                     ┌──────────────────┐
                     │    User Query    │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   Query Router   │
                     │  (LLM Classifier)│
                     └────────┬─────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Domain Expert │      │  Web Search   │      │ General Chat  │
│  (e.g., Code/ │      │    (Tavily)   │      │   Assistant   │
│   Math/Docs)  │      │               │      │               │
└───────┬───────┘      └───────┬───────┘      └───────┬───────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │  Final Response  │
                     └──────────────────┘
```

---

## 📁 Project Structure

```
02_Intelligent_Query_Router/
│
├── intelligent_query_router.ipynb   # Interactive walkthrough & implementation
├── requirements.txt                 # Project dependencies
├── .env                             # Environment variables (API keys)
├── .env.example                     # Example template for environment variables
└── README.md                        # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### 2. Setup Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
# Windows PowerShell
cp .env.example .env
```

Set your API keys inside `.env`:
- `OPENAI_API_KEY` or `GOOGLE_API_KEY`
- `TAVILY_API_KEY` (for search capabilities)
- `LANGCHAIN_API_KEY` (optional, for LangSmith tracing)

### 5. Run the Notebook
Launch Jupyter Notebook / Lab or open in VS Code / Cursor / IDE:
```bash
jupyter notebook intelligent_query_router.ipynb
```

---

## 🛠️ Features
- **Semantic & Intent-Based Routing**: Accurately classifies user queries using structured LLM outputs.
- **Conditional Branching with LangGraph**: Dynamic state transitions without hardcoded nested logic.
- **Fallback Handling**: Gracefully handles ambiguous or unsupported queries.
- **Observability**: Built-in integration with LangSmith for debugging and graph trace visualization.
