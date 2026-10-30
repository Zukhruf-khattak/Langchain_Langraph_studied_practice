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
You are an expert technical interviewer.

Generate interview questions for the following role:

Job Role:
{job_role}

Experience Level:
{experience_level}

Number of Questions:
{number_of_questions}

Focus Area:
{focus_area}

Requirements:
- Generate relevant interview questions.
- Match the difficulty to the experience level.
- Include a mixture of conceptual and practical questions.
- Do not provide answers.
- Number the questions clearly.
- Return only the interview questions.
""",
    input_variables=[
        "job_role",
        "experience_level",
        "number_of_questions",
        "focus_area"
    ]
)


# --------------------------------
# LCEL Chain
# --------------------------------

chain = prompt | model | StrOutputParser()


# --------------------------------
# Generate Questions
# --------------------------------

def generate_questions(
    job_role,
    experience_level,
    number_of_questions,
    focus_area
):

    if not job_role.strip():
        return "Please enter a job role."

    if not focus_area.strip():
        return "Please enter a focus area."

    result = chain.invoke({
        "job_role": job_role,
        "experience_level": experience_level,
        "number_of_questions": number_of_questions,
        "focus_area": focus_area
    })

    return result


# --------------------------------
# Gradio UI
# --------------------------------

with gr.Blocks(title="Interview Question Generator") as app:

    gr.Markdown(
        """
        # Interview Question Generator

        Generate customized interview questions based on
        job role, experience level, and technical focus.
        """
    )

    job_role = gr.Textbox(
        label="Job Role",
        placeholder="e.g. AI Engineer"
    )

    with gr.Row():

        experience_level = gr.Dropdown(
            choices=[
                "Beginner",
                "Intermediate",
                "Advanced"
            ],
            value="Intermediate",
            label="Experience Level"
        )

        number_of_questions = gr.Slider(
            minimum=1,
            maximum=20,
            value=10,
            step=1,
            label="Number of Questions"
        )

    focus_area = gr.Textbox(
        label="Focus Area",
        placeholder="e.g. Python, LangChain, RAG, Machine Learning",
        lines=3
    )

    generate_button = gr.Button(
        "Generate Questions",
        variant="primary"
    )

    output = gr.Textbox(
        label="Interview Questions",
        lines=20
    )

    generate_button.click(
        fn=generate_questions,
        inputs=[
            job_role,
            experience_level,
            number_of_questions,
            focus_area
        ],
        outputs=output
    )


# --------------------------------
# Launch Application
# --------------------------------

if __name__ == "__main__":
    app.launch()