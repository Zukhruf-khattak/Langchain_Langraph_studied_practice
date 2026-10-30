import os
import sys
from pathlib import Path
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
# 1. Configuration & Environment Setup
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("\n[ERROR] GOOGLE_API_KEY is not set in your .env file.")
    print("Please create a .env file with GOOGLE_API_KEY=your_key_here\n")
    sys.exit(1)

DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")
DATA_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Setup Embedding Model (Safe Device Detection)
# --------------------------------------------------

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
except ImportError:
    device = "cpu"

print(f"[*] Embedding computation device: {device.upper()}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)


# --------------------------------------------------
# 3. Vector Store: Cache or Ingestion Pipeline
# --------------------------------------------------

def get_vector_store():
    # Load from local cache if already created
    if INDEX_DIR.exists():
        print(f"[*] Found existing vector index at '{INDEX_DIR}'. Loading cache...")
        try:
            return FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"[!] Failed to load existing index ({e}). Rebuilding...")

    # Otherwise, ingest all PDFs from data/
    pdf_files = sorted(list(DATA_DIR.glob("*.pdf")))
    if not pdf_files:
        print(f"\n[ERROR] No PDF files found inside '{DATA_DIR}/'.")
        print(f"Please place your PDF files in the '{DATA_DIR}/' folder and run again.\n")
        sys.exit(1)

    print(f"[*] Found {len(pdf_files)} PDF file(s) in '{DATA_DIR}':")
    for pdf in pdf_files:
        print(f"    - {pdf.name}")

    all_docs = []
    for pdf_path in pdf_files:
        print(f"[*] Loading: {pdf_path.name}...")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = pdf_path.name
            all_docs.extend(docs)
            print(f"    Loaded {len(docs)} pages from {pdf_path.name}")
        except Exception as e:
            print(f"    [!] Error loading {pdf_path.name}: {e}")

    if not all_docs:
        print("\n[ERROR] Failed to extract text from any PDF files.\n")
        sys.exit(1)

    print(f"\n[*] Total pages loaded: {len(all_docs)}")

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"[*] Created {len(chunks)} text chunks.")

    # Create vector database
    print("[*] Generating embeddings and building FAISS vector store...")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Save cache locally for instant future startups
    vector_store.save_local(str(INDEX_DIR))
    print(f"[*] Vector store saved to '{INDEX_DIR}' for fast reloading.\n")

    return vector_store


vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# --------------------------------------------------
# 4. Create LLM & Conversational Prompt
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful PDF question-answering assistant.

Answer the user's question accurately using ONLY the provided context.
Consider the conversational history for context on follow-up questions.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided PDF(s)."

Context:
{context}"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    (
        "human",
        "{question}"
    )
])

chain = prompt | llm | StrOutputParser()


# --------------------------------------------------
# 5. Core Q&A Function
# --------------------------------------------------

chat_history = []

def ask_question(question: str):
    global chat_history

    retrieved_docs = retriever.invoke(question)

    context_blocks = []
    for doc in retrieved_docs:
        source_file = doc.metadata.get("source_file", "PDF Document")
        page = doc.metadata.get("page", 0) + 1
        context_blocks.append(f"[Source: {source_file} | Page: {page}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_blocks)

    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })

    # Keep conversation history (last 5 turns)
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    return answer, retrieved_docs


# --------------------------------------------------
# 6. Interactive Chat Loop
# --------------------------------------------------

def main():
    print("=" * 60)
    print("       Multi-PDF RAG Question-Answering Bot")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("\nThinking...")
        answer, docs = ask_question(question)
        print(f"\nAssistant: {answer}\n")

        # Display unique sources cited
        sources = set()
        for doc in docs:
            fname = doc.metadata.get("source_file", "PDF Document")
            pnum = doc.metadata.get("page", 0) + 1
            sources.add(f"{fname} (Page {pnum})")

        if sources:
            print("Sources:")
            for s in sorted(sources):
                print(f"  * {s}")


if __name__ == "__main__":
    main()