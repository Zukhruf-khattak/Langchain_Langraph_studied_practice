# LangChain Projects — Comprehensive README

This repository is a collection of **LangChain projects built from fundamentals to Retrieval-Augmented Generation (RAG)**. It documents my hands-on progression with LLM applications, prompt engineering, chains, structured output, conversational memory, and document-based question answering.

The projects are intentionally organized as independent folders so each project can be studied, executed, and explained separately.

```text
LangChain-Projects/
│
├── AI_Text_Summarizer/
├── AI_Email_Rewriter/
├── AI_Study_Notes_Generator/
├── AI_Translator/
├── Chatbot_with_Memory/
├── Interview_Question_Generator/
├── JSON_Information_Extractor/
├── Multi_Style_Text_Generator/
└── Rag_Multi_PDF_QA_Bot/
```

## Projects

| #  | Project                          | Main Concepts                                                  |
| -- | -------------------------------- | -------------------------------------------------------------- |
| 01 | **AI Text Summarizer**           | LLMs, prompts, LCEL, output parsing                            |
| 02 | **AI Email Rewriter**            | Prompt engineering, text transformation, LCEL                  |
| 03 | **AI Study Notes Generator**     | LLM application, structured prompting, educational NLP         |
| 04 | **AI Translator**                | Multilingual NLP, prompt engineering                           |
| 05 | **Chatbot with Memory**          | Conversational AI, message history, memory                     |
| 06 | **Interview Question Generator** | Prompt engineering, controlled generation                      |
| 07 | **JSON Information Extractor**   | Structured output, information extraction                      |
| 08 | **Multi-Style Text Generator**   | Conditional prompting, style control                           |
| 09 | **RAG Multi-PDF QA Bot**         | RAG, PDF loading, chunking, embeddings, FAISS, retrieval, LLMs |

---

## 1. AI Text Summarizer

A simple LangChain application that takes text as input and generates a concise summary using an LLM.

### Concepts

* `ChatGoogleGenerativeAI`
* `PromptTemplate`
* LangChain Expression Language (LCEL)
* `StrOutputParser`
* Prompt engineering
* LLM invocation

### Pipeline

```text
User Text
   ↓
Prompt Template
   ↓
LLM
   ↓
Output Parser
   ↓
Summary
```

---

## 2. AI Email Rewriter

An AI-powered email rewriting application that transforms an existing email according to the required tone or writing style.

### Concepts

* Prompt engineering
* LLMs
* LCEL
* Text transformation
* Tone control

### Pipeline

```text
Original Email
      ↓
Prompt
      ↓
LLM
      ↓
Rewritten Email
```

---

## 3. AI Study Notes Generator

An application that uses an LLM to transform educational content into organized study notes.

### Concepts

* Prompt engineering
* Educational NLP
* LLM-based content generation
* Structured prompts
* Text transformation

### Pipeline

```text
Study Material
      ↓
Prompt Template
      ↓
LLM
      ↓
Structured Study Notes
```

---

## 4. AI Translator

A multilingual translation application built using LangChain and an LLM.

### Concepts

* Natural Language Processing
* Multilingual generation
* Prompt engineering
* LLMs
* Language transformation

### Pipeline

```text
Source Text
    ↓
Source Language
    ↓
Target Language
    ↓
LLM
    ↓
Translated Text
```

---

## 5. Chatbot with Memory

A conversational AI application that maintains previous conversation context instead of treating every user message as an isolated request.

### Concepts

* Conversational AI
* Chat models
* Message history
* Context management
* Prompt templates
* LangChain

### Pipeline

```text
User Message
      ↓
Conversation History
      ↓
Prompt
      ↓
LLM
      ↓
Assistant Response
      ↓
Updated History
```

This project demonstrates an important difference between a basic LLM application and a conversational application: **maintaining context across interactions**.

---

## 6. Interview Question Generator

An AI application that generates interview questions based on the requested topic or technical area.

### Concepts

* Prompt engineering
* Controlled text generation
* LLMs
* Dynamic prompts
* LangChain chains

### Pipeline

```text
Topic / Skill
     ↓
Prompt
     ↓
LLM
     ↓
Interview Questions
```

This project demonstrates how prompts can be designed to control the type and structure of generated content.

---

## 7. JSON Information Extractor

An information extraction application designed to convert unstructured text into structured JSON-style information.

### Concepts

* Information extraction
* Structured output
* Prompt engineering
* LLMs
* JSON generation
* NLP

### Pipeline

```text
Unstructured Text
       ↓
Extraction Prompt
       ↓
LLM
       ↓
Structured JSON
```

This project introduces an important concept for production AI systems: converting natural-language input into structured data that can be consumed by software.

---

## 8. Multi-Style Text Generator

An application that generates text according to different writing styles.

Examples of possible styles include:

* Professional
* Casual
* Formal
* Creative
* Technical

### Concepts

* Prompt engineering
* Conditional prompting
* Style control
* LLM generation
* LCEL

### Pipeline

```text
User Input
    ↓
Selected Style
    ↓
Dynamic Prompt
    ↓
LLM
    ↓
Style-Controlled Output
```

---

# 9. RAG Multi-PDF QA Bot

The most advanced project in this repository is a **Retrieval-Augmented Generation (RAG) application** that allows users to ask questions about information contained in multiple PDF documents.

Instead of relying only on the LLM's pretrained knowledge, the system retrieves relevant information from uploaded documents and uses that information to generate an answer.

### RAG Architecture

```text
                PDF Documents
                     │
                     ▼
              PDF Document Loader
                     │
                     ▼
               Text Extraction
                     │
                     ▼
             Text Chunking
                     │
                     ▼
                Embeddings
                     │
                     ▼
                FAISS Vector DB
                     │
                     │
User Question ───────┤
                     ▼
                 Retrieval
                     │
                     ▼
            Relevant Documents
                     │
                     ▼
                 LLM / Prompt
                     │
                     ▼
                Final Answer
```

### Main Concepts

* Retrieval-Augmented Generation
* PDF document loading
* Document splitting
* Text chunking
* Embeddings
* Vector databases
* FAISS
* Semantic search
* Context retrieval
* Prompt engineering
* LLM generation

### RAG Flow

```text
Documents
    ↓
Load
    ↓
Split into chunks
    ↓
Generate embeddings
    ↓
Store vectors
    ↓
User asks question
    ↓
Convert question to embedding
    ↓
Similarity search
    ↓
Retrieve relevant chunks
    ↓
Send context + question to LLM
    ↓
Generate answer
```

---

# Technology Stack

The exact dependencies vary slightly between projects, but the repository primarily explores the following technologies:

### LangChain

Used for:

* Prompt templates
* Chains
* LLM integration
* Document processing
* Retrieval pipelines
* Output parsing

### Google Gemini

Used as the LLM provider in projects that integrate Google's generative AI models.

### Hugging Face

Used for embedding-related workflows where applicable.

### FAISS

Used for vector similarity search in the RAG project.

### Python

Primary programming language for all projects.

### Gradio

Used for creating simple interactive interfaces where applicable.

---

# Core LangChain Concepts Covered

This repository focuses on learning LangChain progressively.

## 1. Models

Connecting applications to language models.

```text
Application
     ↓
LangChain
     ↓
LLM
```

## 2. Prompt Templates

Creating reusable prompts instead of hard-coding instructions.

```text
Input
 ↓
PromptTemplate
 ↓
LLM
```

## 3. LCEL

LangChain Expression Language allows components to be connected into pipelines.

```text
Prompt → Model → Parser
```

## 4. Output Parsers

Used to transform model responses into usable application output.

## 5. Conversation Memory

Used to maintain context between messages.

## 6. Document Loaders

Used to load external information such as PDFs.

## 7. Text Splitters

Used to divide large documents into smaller chunks.

## 8. Embeddings

Used to represent text as numerical vectors.

## 9. Vector Stores

Used to store and search embeddings.

## 10. Retrieval

Used to find the most relevant document chunks for a user query.

## 11. RAG

Combines retrieval with generation:

```text
Retrieve → Augment → Generate
```

---

# Repository Learning Path

The projects follow an increasing level of complexity.

```text
Level 1
│
├── AI Text Summarizer
├── AI Email Rewriter
└── Multi-Style Text Generator
        │
        ▼
Level 2
│
├── AI Translator
├── AI Study Notes Generator
└── Interview Question Generator
        │
        ▼
Level 3
│
├── JSON Information Extractor
└── Chatbot with Memory
        │
        ▼
Level 4
│
└── RAG Multi-PDF QA Bot
```

The goal is to move from simple LLM calls toward more realistic AI application architectures.

---

# Project Structure

Each project is maintained independently.

Example:

```text
AI_Text_Summarizer/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

A project may contain additional files depending on its implementation.

### Important

Virtual environments and environment variables are intentionally excluded from GitHub.

```text
venv/       → ignored
.env        → ignored
__pycache__ → ignored
```

API keys should **never be committed to the repository**.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Zukhruf-khattak/LangChain-Projects.git
```

Move into the repository:

```bash
cd LangChain-Projects
```

Choose a project:

```bash
cd AI_Text_Summarizer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Projects that use API-based LLMs require an API key.

Create a `.env` file inside the individual project folder:

```text
GOOGLE_API_KEY=your_api_key_here
```

Never commit the `.env` file.

The repository's `.gitignore` files are configured to exclude it.

---

# Running a Project

Each project may have a different entry point.

For example:

```bash
python app.py
```

or:

```bash
python main.py
```

Check the individual project's README for project-specific instructions.

---

# Development Approach

These projects were built incrementally to understand how LangChain components work together rather than treating LangChain as a black box.

The progression is:

```text
LLM
 ↓
Prompt
 ↓
Chain
 ↓
Structured Output
 ↓
Conversation
 ↓
Documents
 ↓
Embeddings
 ↓
Vector Search
 ↓
RAG
```

This progression provides a foundation for building more advanced systems such as:

* AI agents
* Tool-calling applications
* Agentic RAG
* Multi-agent systems
* Autonomous workflows
* Production AI APIs
* AI-powered SaaS applications

---

# Future Development

Planned improvements include expanding the repository toward more advanced LangChain and agentic AI systems.

### Planned areas

* Advanced RAG
* Conversational RAG
* Hybrid search
* Reranking
* Metadata filtering
* Agentic RAG
* LangGraph workflows
* Tool calling
* Function calling
* AI agents
* Multi-agent systems
* FastAPI integration
* Streamlit applications
* Gradio interfaces
* Evaluation and observability
* Production deployment

---

# Learning Objectives

Through these projects, I am building practical experience with:

* Large Language Models
* Generative AI
* LangChain
* Prompt Engineering
* NLP
* RAG
* Vector Databases
* Embeddings
* Semantic Search
* Conversational AI
* Structured Information Extraction
* AI Application Development

The repository is intended to document the progression from **basic LLM applications to retrieval-based and agentic AI systems**.

---

# Security

API keys and credentials should never be stored directly in source code.

Use environment variables:

```text
.env
```

and keep them excluded from version control.

If an API key is accidentally exposed, revoke it immediately and generate a new one.

---

# Author

**Zukhruf Khattak**

AI Engineer | Generative AI | LangChain | RAG | Agentic AI

GitHub: [Zukhruf-khattak](https://github.com/Zukhruf-khattak)

---

# License

This repository is intended primarily for learning, experimentation, and portfolio development.

Individual projects may use different libraries and APIs with their own respective licenses and terms of use.
