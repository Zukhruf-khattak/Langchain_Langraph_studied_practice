# Student Management API

A modern, fast, and lightweight RESTful API built with **FastAPI** for managing student records.

---

## 🚀 Features

- **Full CRUD Operations**: Create, Read, Update, and Delete student records.
- **Data Validation**: Request validation and type safety using Pydantic models.
- **Filtering & Search**: Filter students by major and minimum GPA.
- **Interactive Documentation**: Auto-generated Swagger UI and ReDoc endpoints.
- **Proper HTTP Status Codes & Error Handling**: Returns clean JSON responses and appropriate standard error codes (200, 201, 204, 404, 422).

---

## 📁 Project Structure

```text
01-Student-Management-API/
├── main.py            # Main FastAPI application and API routes
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

---

## 🛠️ Getting Started

### 1. Prerequisites

- Python 3.8+ installed on your system.

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Development Server

```bash
uvicorn main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

---

## 📖 API Documentation

Once the server is running, you can explore the interactive API docs:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📌 API Endpoints

| Method | Endpoint | Description | Status Code |
|---|---|---|---|
| `GET` | `/` | Root health check & documentation links | `200 OK` |
| `GET` | `/students` | Get all students (supports filtering by `major`, `min_gpa`) | `200 OK` |
| `GET` | `/students/{student_id}` | Get details of a single student by ID | `200 OK` / `404 Not Found` |
| `POST` | `/students` | Register a new student | `201 Created` |
| `PUT` | `/students/{student_id}` | Update student details (partial or full) | `200 OK` / `404 Not Found` |
| `DELETE` | `/students/{student_id}` | Remove a student record | `204 No Content` / `404 Not Found` |

---

## 📝 Request & Response Examples

### 1. Create a Student (`POST /students`)

**Request Body:**
```json
{
  "name": "Jane Doe",
  "age": 20,
  "email": "jane.doe@example.com",
  "major": "Data Science",
  "gpa": 3.9
}
```

**Response (`201 Created`):**
```json
{
  "id": 3,
  "name": "Jane Doe",
  "age": 20,
  "email": "jane.doe@example.com",
  "major": "Data Science",
  "gpa": 3.9
}
```

### 2. Update a Student (`PUT /students/{student_id}`)

**Request Body:**
```json
{
  "gpa": 4.0
}
```

**Response (`200 OK`):**
```json
{
  "id": 3,
  "name": "Jane Doe",
  "age": 20,
  "email": "jane.doe@example.com",
  "major": "Data Science",
  "gpa": 4.0
}
```
