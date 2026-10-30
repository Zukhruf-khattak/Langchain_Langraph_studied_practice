from dotenv import load_dotenv
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()


# ==========================================
# Gemini Model
# ==========================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# ==========================================
# Prompt Template
# ==========================================

prompt = PromptTemplate(
    template="""
You are an expert email rewriting assistant.

Rewrite the user's email according to the selected tone.

Selected Tone:
{tone}

Rules:
- Keep the original meaning.
- Do not invent information.
- Correct grammar and spelling.
- Make the email clear and natural.
- Do not add unnecessary details.
- Return only the rewritten email.
- Do not explain what you changed.

Original Email:
{email}

Rewritten Email:
""",
    input_variables=["tone", "email"]
)


# ==========================================
# Output Parser
# ==========================================

parser = StrOutputParser()


# ==========================================
# LangChain LCEL Chain
# ==========================================

chain = prompt | model | parser


# ==========================================
# Email Rewriting Function
# ==========================================

def rewrite_email(email, tone):

    if not email.strip():
        return "Please enter an email."

    result = chain.invoke({
        "email": email,
        "tone": tone
    })

    return result


# ==========================================
# Gradio Interface
# ==========================================

interface = gr.Interface(
    fn=rewrite_email,

    inputs=[
        gr.Textbox(
            lines=12,
            placeholder="Paste your email here...",
            label="Original Email"
        ),

        gr.Dropdown(
            choices=[
                "Professional",
                "Friendly",
                "Formal",
                "Concise"
            ],
            value="Professional",
            label="Email Tone"
        )
    ],

    outputs=gr.Textbox(
        lines=12,
        label="Rewritten Email"
    ),

    title="AI Email Rewriter",

    description=(
        "Rewrite emails using LangChain and Gemini. "
        "Choose a tone and generate an improved version."
    )
)


# ==========================================
# Launch Application
# ==========================================

if __name__ == "__main__":
    interface.launch()