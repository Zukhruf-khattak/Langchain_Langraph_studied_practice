from typing import List, Optional
from fastapi import FastAPI, HTTPException, Path, Query, status
from pydantic import BaseModel, Field, EmailStr

app = FastAPI(
    title="Student Management API",
    description="A modern RESTful API for managing student records using FastAPI.",
    version="1.0.0",
)


# --- Models ---

class StudentBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, example="John Doe")
    age: int = Field(..., ge=1, le=120, example=20)
    email: str = Field(..., example="john.doe@example.com")
    major: str = Field(..., min_length=2, max_length=100, example="Computer Science")
    gpa: float = Field(..., ge=0.0, le=4.0, example=3.75)


class StudentCreate(StudentBase):
    pass


class StudentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100, example="John Doe")
    age: Optional[int] = Field(None, ge=1, le=120, example=21)
    email: Optional[str] = Field(None, example="john.updated@example.com")
    major: Optional[str] = Field(None, min_length=2, max_length=100, example="Software Engineering")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, example=3.9)


class StudentResponse(StudentBase):
    id: int = Field(..., example=1)

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "age": 20,
                "email": "john.doe@example.com",
                "major": "Computer Science",
                "gpa": 3.75,
            }
        }


# --- In-Memory Database ---

students_db: dict[int, StudentResponse] = {
    1: StudentResponse(
        id=1,
        name="Alice Johnson",
        age=21,
        email="alice.johnson@example.com",
        major="Computer Science",
        gpa=3.85,
    ),
    2: StudentResponse(
        id=2,
        name="Bob Smith",
        age=22,
        email="bob.smith@example.com",
        major="Electrical Engineering",
        gpa=3.40,
    ),
}

current_id_counter: int = 2


# --- Routes ---

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Welcome to the Student Management API!",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get(
    "/students",
    response_model=List[StudentResponse],
    tags=["Students"],
    summary="Get all students",
)
def get_students(
    major: Optional[str] = Query(None, description="Filter students by major"),
    min_gpa: Optional[float] = Query(None, ge=0.0, le=4.0, description="Filter students with GPA >= min_gpa"),
):
    results = list(students_db.values())
    if major:
        results = [s for s in results if s.major.lower() == major.lower()]
    if min_gpa is not None:
        results = [s for s in results if s.gpa >= min_gpa]
    return results


@app.get(
    "/students/{student_id}",
    response_model=StudentResponse,
    tags=["Students"],
    summary="Get a student by ID",
)
def get_student(
    student_id: int = Path(..., ge=1, description="The ID of the student to retrieve")
):
    if student_id not in students_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID {student_id} not found",
        )
    return students_db[student_id]


@app.post(
    "/students",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Students"],
    summary="Create a new student",
)
def create_student(student: StudentCreate):
    global current_id_counter
    current_id_counter += 1
    new_student = StudentResponse(id=current_id_counter, **student.model_dump())
    students_db[current_id_counter] = new_student
    return new_student


@app.put(
    "/students/{student_id}",
    response_model=StudentResponse,
    tags=["Students"],
    summary="Update an existing student",
)
def update_student(
    student_id: int = Path(..., ge=1, description="The ID of the student to update"),
    student_update: StudentUpdate = ...,
):
    if student_id not in students_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID {student_id} not found",
        )

    stored_student = students_db[student_id]
    update_data = student_update.model_dump(exclude_unset=True)
    updated_student = stored_student.model_copy(update=update_data)
    students_db[student_id] = updated_student
    return updated_student


@app.delete(
    "/students/{student_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Students"],
    summary="Delete a student",
)
def delete_student(
    student_id: int = Path(..., ge=1, description="The ID of the student to delete")
):
    if student_id not in students_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID {student_id} not found",
        )
    del students_db[student_id]
    return None
