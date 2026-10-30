# 🤖 03-AI-Agent-API

A production-ready **FastAPI AI Agent backend** equipped with dynamic tool calling (Calculator, Date/Time, Text Analyzer) powered by Google Gemini.

---

## 📁 Project Structure

```
03-AI-Agent-API/
│
├── app/
│   ├── main.py                # FastAPI application entry point & routes
│   │
│   ├── agents/
│   │   └── agent.py           # AI Agent class with Gemini tool-calling loop
│   │
│   ├── tools/
│   │   ├── calculator.py      # Safe AST-based math evaluation tool
│   │   ├── date_tool.py       # Timezone-aware date/time calculations
│   │   └── text_analyzer.py   # Text statistics & keyword analyzer
│   │
│   ├── schemas/
│   │   └── chat.py            # Pydantic models for chat requests/responses
│   │
│   └── config/
│       └── settings.py        # Pydantic BaseSettings environment config
│
├── .env                       # Local environment variables
├── .env.example               # Example environment variable template
├── .gitignore                 # Git ignore configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get one free on Google AI Studio](https://aistudio.google.com/))

### 2. Installation

Clone or navigate to the project directory:
```bash
cd 02-AI-Agent-API
```

Create and activate a virtual environment:
```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and set your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.5-flash
APP_NAME="AI Agent API"
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

### 4. Running the Server

Start the FastAPI application with Uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

---

## 📖 API Documentation & Endpoints

Interactive Swagger API docs are accessible at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check and API key status |
| `GET` | `/api/v1/tools` | List registered tools and parameter schemas |
| `POST` | `/api/v1/chat` | Send prompt to AI Agent with multi-tool execution |

---

## 🛠️ Available Agent Tools

1. **Calculator (`calculate`)**:
   - Safely evaluates arithmetic and math expressions (e.g. `sqrt(144) * 3`, `2**10`, `sin(pi/2)`).
2. **Date Tools**:
   - `get_current_datetime`: Returns current time for any timezone (e.g. `UTC`, `America/New_York`, `Asia/Kolkata`).
   - `calculate_date_offset`: Calculates future/past dates with day/hour offsets.
   - `get_date_difference`: Computes duration between two dates.
3. **Text Analyzer (`analyze_text`)**:
   - Calculates word count, character count, sentence count, estimated reading duration, and frequent keywords.

---

## 🧪 Example API Requests

### 1. Chat Request (Triggering Calculator & Date Tool)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What is 144 * 25, and what will the date be in 45 days from now in UTC?"
     }'
```

### Response Example
```json
{
  "reply": "The calculation 144 * 25 equals 3,600. In 45 days from today (UTC), the date will be...",
  "tool_calls": [
    {
      "tool_name": "calculate",
      "arguments": { "expression": "144 * 25" },
      "result": "3600"
    },
    {
      "tool_name": "calculate_date_offset",
      "arguments": { "days": 45, "timezone_name": "UTC" },
      "result": "Target DateTime: 2026-10-18 21:55:00 UTC\nOffset: 45 days, 0 hours\nDay of the week: Sunday"
    }
  ],
  "model": "gemini-2.5-flash",
  "execution_time_seconds": 1.1523
}
```

---

## 🛡️ Security
- **Safe Calculator**: Uses Python's Abstract Syntax Tree (`ast`) parser to disallow execution of arbitrary code or injection vulnerabilities.
- **Environment Isolation**: Sensitive credentials are kept in `.env` and excluded from source control via `.gitignore`.
