from dotenv import load_dotenv
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv()


# -----------------------------
# 1. Create the model
# -----------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7
)


# -----------------------------
# 2. Conversation memory
# -----------------------------

chat_history = [
    SystemMessage(
        content=(
            "You are a helpful AI assistant. "
            "Remember previous messages and use them "
            "when answering new questions."
        )
    )
]


# -----------------------------
# 3. Chat function
# -----------------------------

def chat(user_message, history):

    if not user_message.strip():
        return history, ""

    # Add user message to LangChain memory
    chat_history.append(
        HumanMessage(content=user_message)
    )

    try:

        # Send conversation history to Gemini
        response = model.invoke(chat_history)

        ai_response = response.content

        # Add AI response to memory
        chat_history.append(
            AIMessage(content=ai_response)
        )

        # Gradio 6 message format
        history.append({
            "role": "user",
            "content": user_message
        })

        history.append({
            "role": "assistant",
            "content": ai_response
        })

        return history, ""

    except Exception as e:

        # Remove user message if model fails
        chat_history.pop()

        return history, f"Error: {str(e)}"


# -----------------------------
# 4. Clear conversation
# -----------------------------

def clear_chat():

    global chat_history

    chat_history = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant. "
                "Remember previous messages and use them "
                "when answering new questions."
            )
        )
    ]

    return [], ""


# -----------------------------
# 5. Gradio UI
# -----------------------------

with gr.Blocks(title="Chatbot with Memory") as demo:

    gr.Markdown(
        """
        # Chatbot with Memory

        A LangChain chatbot with conversation memory.
        """
    )

    chatbot = gr.Chatbot(
        label="AI Assistant"
    )

    message = gr.Textbox(
        label="Your Message",
        placeholder="Type your message..."
    )

    with gr.Row():

        send_button = gr.Button(
            "Send",
            variant="primary"
        )

        clear_button = gr.Button("Clear Chat")


    # Send button
    send_button.click(
        fn=chat,
        inputs=[message, chatbot],
        outputs=[chatbot, message]
    )


    # Enter key
    message.submit(
        fn=chat,
        inputs=[message, chatbot],
        outputs=[chatbot, message]
    )


    # Clear button
    clear_button.click(
        fn=clear_chat,
        outputs=[chatbot, message]
    )


# -----------------------------
# 6. Launch
# -----------------------------

if __name__ == "__main__":
    demo.launch()