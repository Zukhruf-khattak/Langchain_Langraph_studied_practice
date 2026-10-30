import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


load_dotenv()


# Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# Conversation history
chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]


def chat(message, history):
    # Add user message
    chat_history.append(
        HumanMessage(content=message)
    )

    # Get Gemini response
    result = model.invoke(chat_history)

    # Add AI response
    chat_history.append(
        AIMessage(content=result.content)
    )

    return result.content


# GUI
demo = gr.ChatInterface(
    fn=chat,
    title="Gemini AI Chatbot",
    description="Chat with Gemini using LangChain",
    textbox=gr.Textbox(
        placeholder="Type your message...",
        container=True
    )
)


if __name__ == "__main__":
    demo.launch()