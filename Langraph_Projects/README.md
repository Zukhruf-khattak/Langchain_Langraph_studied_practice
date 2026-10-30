# 🦜🕸️ LangGraph Projects & Design Patterns

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.11%2B-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-1.3%2B-green.svg)](https://github.com/langchain-ai/langchain)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A curated collection of practical, modular, and production-ready **LangGraph** workflows and multi-agent design patterns built with **LangChain**, **Google Gemini**, and **Python**.

---

## 📂 Projects & Architecture Patterns

| # | Project | Pattern | Description | Key Tech | Link |
|---|---|---|---|---|---|
| **01** | **Sequential AI Workflow** | `Linear Chain` | Deterministic pipeline for sequential text processing, summarization, and tone enhancement. | LangGraph, Gemini 3.7 Flash, TypedDict | [Explore](./01_Sequential_AI_Workflow) |
| **02** | **Intelligent Query Router** | `Conditional Routing` | Dynamic intent classification and semantic routing to specialized expert handlers or search tools. | LangGraph, Conditional Edges, LangChain | [Explore](./02_Intelligent_Query_Router) |
| **03** | **Parallel Content Analyzer** | `Fan-Out / Fan-In` | Concurrent execution of sentiment, keyword extraction, and summarization branches with synchronized report synthesis. | LangGraph, Concurrent Edges, TypedDict | [Explore](./03_Parallel_Content_Analyzer) |
| **04** | **Iterative Answer Improvement** | `Reflection & Cyclic Loop` | Self-correcting feedback loop with structured critique scoring (1–10) and threshold-based stopping. | LangGraph, Cyclic Graphs, Pydantic, CLI | [Explore](./04_Iterative_Answer_Improvement) |

---

## 🏗️ Architectural Patterns Comparison

```
1. Sequential Chain                 2. Conditional Router
   ┌────────┐   ┌────────┐              ┌────────┐   ┌────────────────┐
   │ Node 1 │──►│ Node 2 │              │ Router │──►│ Expert A / B / C│
   └────────┘   └────────┘              └────────┘   └────────────────┘

3. Parallel Fan-Out / Fan-In        4. Cyclic Reflection Loop
        ┌──► Branch A ──┐                  ┌──────────┐     ┌────────┐
        ├──► Branch B ──┼──► Merge         │ Generate │────►│ Critic │◄───┐
        └──► Branch C ──┘                  └──────────┘     └───┬────┘    │
                                                                │ [Score] │
                                                                ▼         │
                                                            [Pass?] ──No──┘
                                                                │ Yes
                                                                ▼
                                                             [ END ]
```

---

## 📁 Repository Structure

```text
LangraphProjects/
├── .gitignore
├── README.md
│
├── 01_Sequential_AI_Workflow/
│   ├── sequential_workflow.ipynb        # Jupyter notebook walkthrough
│   ├── requirements.txt                 # Dependencies
│   ├── .env.example                     # Environment template
│   └── README.md                        # Project documentation
│
├── 02_Intelligent_Query_Router/
│   ├── intelligent_query_router.ipynb   # Jupyter notebook walkthrough
│   ├── PROJECT_OVERVIEW.md              # Detailed architecture breakdown
│   ├── requirements.txt                 # Dependencies
│   ├── .env.example                     # Environment template
│   └── README.md                        # Project documentation
│
├── 03_Parallel_Content_Analyzer/
│   ├── parallel_content_analyzer.ipynb  # Jupyter notebook walkthrough
│   ├── requirements.txt                 # Dependencies
│   ├── .env.example                     # Environment template
│   └── README.md                        # Project documentation
│
└── 04_Iterative_Answer_Improvement/
    ├── main.py                          # CLI executable application
    ├── iterative_answer_improvement.ipynb # Jupyter notebook walkthrough
    ├── requirements.txt                 # Dependencies
    ├── .env.example                     # Environment template
    └── README.md                        # Project documentation
```

---

## ⚡ Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Zukhruf-khattak/Langraph_Projects.git
cd Langraph_Projects
```

### 2. Navigate to Any Project Directory
```bash
cd 01_Sequential_AI_Workflow
# or
cd 02_Intelligent_Query_Router
# or
cd 03_Parallel_Content_Analyzer
# or
cd 04_Iterative_Answer_Improvement
```

### 3. Create & Activate a Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Environment Variables
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 6. Run the Project
- **For Jupyter Notebooks**:
  ```bash
  jupyter notebook
  ```
- **For 04 CLI Script**:
  ```bash
  python main.py
  ```

---

## 🛠️ Core LangGraph Capabilities Demonstrated

- **`StateGraph` Architecture**: Composable graph pipelines with typed state containers (`TypedDict`).
- **Control Flow & Edges**:
  - Linear Edges (`add_edge(START, node)`)
  - Parallel Branching (`builder.add_edge(START, node_a)`, `builder.add_edge(START, node_b)`)
  - Conditional Edges (`add_conditional_edges`)
  - Cyclic Graphs & Self-Correction Loops
- **Structured LLM Outputs**: Utilizing Pydantic models with `with_structured_output()` for strict schema enforcement.
- **Execution Paradigms**:
  - `app.invoke()` for complete synchronous execution.
  - `app.stream()` for real-time node-by-node telemetry and state inspection.
- **Graph Topology Visualization**: Dynamic Mermaid rendering (`draw_mermaid_png()`).

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
