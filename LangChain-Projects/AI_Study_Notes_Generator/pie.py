from dotenv import load_dotenv
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------
# Load Environment Variables
# --------------------------------

load_dotenv()


# --------------------------------
# Initialize Gemini
# --------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# --------------------------------
# Prompt Template
# --------------------------------

prompt = PromptTemplate(
    template="""
You are an expert educational content creator.

Create clear and useful study notes for the following topic.

Subject:
{subject}

Topic:
{topic}

Difficulty Level:
{difficulty}

Note Length:
{note_length}

Create the notes using exactly this structure:

# Topic

## 1. Overview
Give a simple explanation of the topic.

## 2. Key Concepts
Explain the most important concepts.

## 3. Important Terms
List important terms with short definitions.

## 4. Examples
Give practical examples where appropriate.

## 5. Key Points to Remember
List the most important points for revision.

## 6. Quick Revision
Create a short revision section for a student.

Requirements:
- Match the explanation to the difficulty level.
- Use simple and clear language.
- Avoid unnecessary information.
- Use bullet points where appropriate.
- Make the notes useful for exam preparation.
- Return only the study notes.
""",
    input_variables=[
        "subject",
        "topic",
        "difficulty",
        "note_length"
    ]
)


# --------------------------------
# LCEL Chain
# --------------------------------

chain = prompt | model | StrOutputParser()


# --------------------------------
# Generate Study Notes
# --------------------------------

def generate_notes(
    subject,
    topic,
    difficulty,
    note_length
):

    if not subject.strip():
        return "Please enter a subject."

    if not topic.strip():
        return "Please enter a topic."

    result = chain.invoke({
        "subject": subject,
        "topic": topic,
        "difficulty": difficulty,
        "note_length": note_length
    })

    return result


# --------------------------------
# Gradio UI
# --------------------------------

with gr.Blocks(title="AI Study Notes Generator") as app:

    gr.Markdown(
        """
        # AI Study Notes Generator

        Generate structured study notes using LangChain and Gemini.
        """
    )

    subject = gr.Textbox(
        label="Subject",
        placeholder="e.g. Artificial Intelligence"
    )

    topic = gr.Textbox(
        label="Topic",
        placeholder="e.g. Neural Networks",
        lines=3
    )

    with gr.Row():

        difficulty = gr.Dropdown(
            choices=[
                "Beginner",
                "Intermediate",
                "Advanced"
            ],
            value="Intermediate",
            label="Difficulty Level"
        )

        note_length = gr.Dropdown(
            choices=[
                "Short",
                "Medium",
                "Detailed"
            ],
            value="Medium",
            label="Note Length"
        )

    generate_button = gr.Button(
        "Generate Study Notes",
        variant="primary"
    )

    output = gr.Markdown(
        label="Study Notes"
    )

    generate_button.click(
        fn=generate_notes,
        inputs=[
            subject,
            topic,
            difficulty,
            note_length
        ],
        outputs=output
    )


# --------------------------------
# Launch
# --------------------------------

if __name__ == "__main__":
    app.launch()