# 🚀 FastAPI Projects & Practice Repository

A curated collection of practical, production-ready RESTful APIs and Intelligent Agent backends built with **FastAPI**, **Pydantic**, and **Python**.

---

## 📂 Projects in this Repository

| # | Project | Description | Key Technologies | Link |
|---|---|---|---|---|
| **01** | **Student Management API** | Full CRUD RESTful API for managing student records with Pydantic validation, search/filtering, and automatic docs. | FastAPI, Pydantic, Uvicorn | [View Project](./01-Student-Management-API) |
| **02** | **AI Agent API** | Intelligent conversational AI backend equipped with dynamic tool calling (Calculator, Date/Time, Text Analyzer) and LLM integration. | FastAPI, Google Gemini, Pydantic, Python-dotenv | [View Project](./02-AI-Agent-API) |

---

## 🛠️ Repository Structure

```text
FastAPI-Projects/
├── .gitignore
├── README.md
├── 01-Student-Management-API/
│   ├── main.py                  # API routes and CRUD logic
│   ├── requirements.txt         # Dependencies
│   └── README.md                # Project-specific documentation
└── 02-AI-Agent-API/
    ├── app/
    │   ├── main.py              # FastAPI app initialization & endpoints
    │   ├── agents/              # AI agent logic and conversation handling
    │   ├── config/              # Application settings and environment config
    │   ├── schemas/             # Pydantic request and response schemas
    │   └── tools/               # Agent tool implementations (calculator, date, text)
    ├── .env.example             # Example environment variables
    ├── requirements.txt         # Dependencies
    └── README.md                # Project-specific documentation
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Zukhruf-khattak/FastAPIPractice.git
cd FastAPIPractice
```

### 2. Choose a Project

#### 📌 Option A: Run 01-Student-Management-API

```bash
cd 01-Student-Management-API

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies and run
pip install -r requirements.txt
uvicorn main:app --reload
```
Interactive Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

#### 🤖 Option B: Run 02-AI-Agent-API

```bash
cd 02-AI-Agent-API

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Add your GEMINI_API_KEY in .env

# Run server
uvicorn app.main:app --reload
```
Interactive Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📖 Technologies & Concepts Covered

- **FastAPI Core**: Request handling, routing, path/query parameters, status codes, response models.
- **Pydantic Validation**: Strong typing, schema definitions, input sanitization, error serialization.
- **AI & Tool Execution**: LLM integration, function calling, agentic tool workflows.
- **API Documentation**: Automated OpenAPI Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **Clean Architecture**: Modular folder structures, separation of concerns, configuration management.

---

## 👤 Author

- GitHub: [@Zukhruf-khattak](https://github.com/Zukhruf-khattak)
