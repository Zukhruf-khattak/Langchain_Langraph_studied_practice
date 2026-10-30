from dotenv import load_dotenv
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

prompt = PromptTemplate(
    template="""
You are an expert text summarizer.

Summarize the following text into 4 to 6 clear bullet points.

Rules:
- Each bullet should contain one important idea.
- Use simple language.
- Do not add information that is not present in the text.

Text:
{text}

Bullet Point Summary:
""",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt | model | parser


def summarize_text(text):
    if not text.strip():
        return "Please enter some text to summarize."

    result = chain.invoke({
        "text": text
    })

    return result


interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        lines=12,
        placeholder="Paste your text here...",
        label="Text to Summarize"
    ),
    outputs=gr.Textbox(
        lines=8,
        label="Bullet Point Summary"
    ),
    title="AI Text Summarizer",
    description="Summarize text into clear and concise bullet points using LangChain and Gemini."
)

interface.launch()