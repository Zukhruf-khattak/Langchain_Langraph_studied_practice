from dotenv import load_dotenv
import gradio as gr

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


# --------------------------------
# Load Environment Variables
# --------------------------------

load_dotenv()


# --------------------------------
# Structured Output Schema
# --------------------------------

class PersonInformation(BaseModel):

    name: str = Field(
        description="Full name of the person"
    )

    age: int | None = Field(
        default=None,
        description="Age of the person if available"
    )

    profession: str | None = Field(
        default=None,
        description="Profession or job of the person"
    )

    location: str | None = Field(
        default=None,
        description="Location of the person"
    )

    skills: list[str] = Field(
        default_factory=list,
        description="Skills mentioned in the text"
    )


# --------------------------------
# Initialize Gemini
# --------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# --------------------------------
# Structured Model
# --------------------------------

structured_model = model.with_structured_output(
    PersonInformation
)


# --------------------------------
# Prompt Template
# --------------------------------

prompt = PromptTemplate(
    template="""
Extract information about the person from the following text.

Return the information according to the required structured schema.

Text:
{text}

Rules:
- Extract only information present in the text.
- Do not invent information.
- If information is missing, use null or an empty list.
""",
    input_variables=["text"]
)


# --------------------------------
# LCEL Chain
# --------------------------------

chain = prompt | structured_model


# --------------------------------
# Extraction Function
# --------------------------------

def extract_information(text):

    if not text.strip():
        return "Please enter some text."

    try:

        result = chain.invoke({
            "text": text
        })

        return result.model_dump_json(
            indent=2
        )

    except Exception as e:

        return f"Error: {str(e)}"


# --------------------------------
# Gradio UI
# --------------------------------

with gr.Blocks(
    title="JSON Information Extractor"
) as app:

    gr.Markdown(
        """
        # JSON Information Extractor

        Extract structured information from unstructured text
        using LangChain, Gemini, and Pydantic.
        """
    )

    text_input = gr.Textbox(
        label="Input Text",
        placeholder="Enter information about a person...",
        lines=10
    )

    extract_button = gr.Button(
        "Extract Information",
        variant="primary"
    )

    output = gr.Code(
        label="Extracted JSON",
        language="json"
    )

    extract_button.click(
        fn=extract_information,
        inputs=text_input,
        outputs=output
    )


# --------------------------------
# Launch
# --------------------------------

if __name__ == "__main__":
    app.launch()