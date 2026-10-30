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
# Translation Prompt
# --------------------------------

prompt = PromptTemplate(
    template="""
You are a professional translator.

Translate the following text from {source_language}
to {target_language}.

Rules:
- Preserve the original meaning.
- Keep the translation natural and fluent.
- Preserve the original tone.
- Do not add explanations.
- Return only the translated text.

Text:
{text}
""",
    input_variables=[
        "source_language",
        "target_language",
        "text"
    ]
)


# --------------------------------
# LCEL Chain
# --------------------------------

chain = prompt | model | StrOutputParser()


# --------------------------------
# Translation Function
# --------------------------------

def translate_text(text, source_language, target_language):

    if not text.strip():
        return "Please enter some text."

    if source_language == target_language:
        return "Source and target languages must be different."

    result = chain.invoke({
        "source_language": source_language,
        "target_language": target_language,
        "text": text
    })

    return result


# --------------------------------
# Gradio UI
# --------------------------------

with gr.Blocks(title="AI Translator") as app:

    gr.Markdown(
        """
        # AI Translator

        Translate text between different languages using
        LangChain and Gemini.
        """
    )

    with gr.Row():

        source_language = gr.Dropdown(
            choices=[
                "English",
                "Urdu",
                "Arabic",
                "Spanish",
                "French",
                "German",
                "Chinese",
                "Japanese",
                "Hindi"
            ],
            value="English",
            label="Source Language"
        )

        target_language = gr.Dropdown(
            choices=[
                "English",
                "Urdu",
                "Arabic",
                "Spanish",
                "French",
                "German",
                "Chinese",
                "Japanese",
                "Hindi"
            ],
            value="Urdu",
            label="Target Language"
        )

    text = gr.Textbox(
        label="Text",
        placeholder="Enter text to translate...",
        lines=8
    )

    translate_button = gr.Button(
        "Translate",
        variant="primary"
    )

    output = gr.Textbox(
        label="Translation",
        lines=8
    )

    translate_button.click(
        fn=translate_text,
        inputs=[
            text,
            source_language,
            target_language
        ],
        outputs=output
    )


# --------------------------------
# Launch
# --------------------------------

if __name__ == "__main__":
    app.launch()