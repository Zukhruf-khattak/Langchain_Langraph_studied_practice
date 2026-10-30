from dotenv import load_dotenv

import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()


# --------------------------------
# 1. Initialize Gemini
# --------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.8
)


# --------------------------------
# 2. Prompt Template
# --------------------------------

prompt = PromptTemplate(
    template="""
You are a professional multi-style text generation assistant.

Generate high-quality text based on the topic and requested style.

Topic:
{topic}

Style:
{style}

Requirements:
- Follow the requested style carefully.
- Keep the content relevant to the topic.
- Make the writing natural and engaging.
- Do not explain the style.
- Return only the generated text.
""",
    input_variables=["topic", "style"]
)


# --------------------------------
# 3. LCEL Chain
# --------------------------------

chain = prompt | model | StrOutputParser()


# --------------------------------
# 4. Generate Text Function
# --------------------------------

def generate_text(topic, style):

    if not topic.strip():
        return "Please enter a topic."

    if not style:
        return "Please select a style."

    result = chain.invoke({
        "topic": topic,
        "style": style
    })

    return result


# --------------------------------
# 5. Gradio UI
# --------------------------------

with gr.Blocks(title="Multi-Style Text Generator") as app:

    gr.Markdown(
        """
        # Multi-Style Text Generator

        Generate text in different writing styles using LangChain and Gemini.
        """
    )

    topic = gr.Textbox(
        label="Topic",
        placeholder="Enter your topic...",
        lines=4
    )

    style = gr.Dropdown(
        choices=[
            "Professional",
            "Casual",
            "Creative",
            "Academic",
            "Social Media"
        ],
        label="Writing Style",
        value="Professional"
    )

    generate_button = gr.Button(
        "Generate Text",
        variant="primary"
    )

    output = gr.Textbox(
        label="Generated Text",
        lines=12
    )

    generate_button.click(
        fn=generate_text,
        inputs=[topic, style],
        outputs=output
    )


# --------------------------------
# 6. Launch App
# --------------------------------

if __name__ == "__main__":
    app.launch()