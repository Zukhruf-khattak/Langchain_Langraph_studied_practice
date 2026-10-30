import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")
DATA_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. API Key Validation
# --------------------------------------------------

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ **GOOGLE_API_KEY is not configured.** Please add it to your `.env` file or environment variables.")
    st.stop()


# --------------------------------------------------
# 3. Model Initialization (Cached)
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading Embedding Model & LLM...")
def initialize_models():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )
    return embeddings, llm, device


embeddings, llm, device = initialize_models()


# --------------------------------------------------
# 4. Session State Management
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    # Try loading existing FAISS index on startup
    if INDEX_DIR.exists():
        try:
            st.session_state.vector_store = FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None


# --------------------------------------------------
# 5. Helper Functions: Ingestion & Q&A
# --------------------------------------------------

def build_vector_store():
    pdf_files = sorted(list(DATA_DIR.glob("*.pdf")))
    if not pdf_files:
        return None, "No PDF files found in `data/` directory."

    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = pdf_path.name
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"Could not load {pdf_path.name}: {e}")

    if not all_docs:
        return None, "Could not extract text from the PDF files."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)

    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(str(INDEX_DIR))
    return vector_store, f"Indexed {len(pdf_files)} PDF(s) ({len(all_docs)} pages, {len(chunks)} chunks)."


# Prompt & Chain
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful Multi-PDF Question Answering Assistant.

Answer the user's question using ONLY the provided retrieved context.
If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided PDFs."

Retrieved Context:
{context}"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    (
        "human",
        "{question}"
    )
])

chain = prompt | llm | StrOutputParser()


def ask_question(question: str):
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)

    context_blocks = []
    for doc in retrieved_docs:
        source_file = doc.metadata.get("source_file", "PDF Document")
        page = doc.metadata.get("page", 0) + 1
        context_blocks.append(f"[Source: {source_file} | Page: {page}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_blocks)

    answer = chain.invoke({
        "context": context,
        "chat_history": st.session_state.chat_history[-6:],
        "question": question
    })

    return answer, retrieved_docs


# --------------------------------------------------
# 6. Sidebar UI
# --------------------------------------------------

with st.sidebar:
    st.title("📚 Knowledge Base")
    st.caption(f"Hardware Device: **{device.upper()}**")

    st.markdown("### 📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = DATA_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to `data/`")

    # List current files in data/
    current_pdfs = sorted(list(DATA_DIR.glob("*.pdf")))
    st.markdown(f"### 📄 Files in `data/` ({len(current_pdfs)})")
    if current_pdfs:
        for pdf in current_pdfs:
            st.text(f"• {pdf.name}")
    else:
        st.info("No PDF files in `data/`.")

    st.divider()

    # Build / Rebuild Button
    if st.button("🔨 Build / Rebuild Knowledge Base", use_container_width=True, type="primary"):
        with st.spinner("Processing PDFs and generating embeddings..."):
            v_store, msg = build_vector_store()
            if v_store:
                st.session_state.vector_store = v_store
                st.session_state.chat_history = []
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

    # Clear Index
    if st.button("🗑️ Clear Vector Index", use_container_width=True):
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.info("Vector index cleared.")
        st.rerun()

    st.divider()
    if st.session_state.vector_store is not None:
        st.success("🟢 Knowledge Base Active")
    else:
        st.warning("🔴 Knowledge Base Not Loaded")


# --------------------------------------------------
# 7. Main Chat UI
# --------------------------------------------------

st.title("💬 Multi-PDF RAG Assistant")
st.caption("Ask questions across all your PDF documents with source references and conversational memory.")

# Display Chat History
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Prompt for Question
if st.session_state.vector_store is None:
    st.info("👉 Click **Build Knowledge Base** in the sidebar to index your PDFs and start chatting.")
else:
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Display user question
        with st.chat_message("user"):
            st.write(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating answer..."):
                answer, docs = ask_question(user_input)
            st.write(answer)

            # Display source expander
            if docs:
                sources = set()
                for doc in docs:
                    fname = doc.metadata.get("source_file", "PDF Document")
                    pnum = doc.metadata.get("page", 0) + 1
                    sources.add(f"{fname} (Page {pnum})")

                with st.expander("🔍 View Referenced Sources"):
                    for s in sorted(sources):
                        st.markdown(f"- 📄 `{s}`")

        # Save to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))
