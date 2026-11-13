"""
rag_pipeline.py - Core RAG and Guardrail Pipeline for Zyro Dynamics HR Assistant.

Preserves the exact architecture, retrieval logic, prompts, chunking, and guardrails.
Supports high-speed free Groq models (openai/gpt-oss-120b, openai/gpt-oss-20b)
as well as Google GenAI (gemini-3.7-flash).
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM Providers
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv(override=True)

# --- Original Notebook Constants ---
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
DEFAULT_TOP_K = 5
CORPUS_PATH = "./zyro-dynamics-hr-corpus"
VECTORSTORE_DIR = "./vectorstore/faiss_index"

# --- RAG Prompt Template (Optimized for thorough, informative policy answers) ---
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an HR assistant for Zyro Dynamics. Based on the company HR policy context provided below, provide a clear, helpful, comprehensive, and detailed answer to the employee's inquiry. Summarize and explain all relevant rules, entitlements, procedures, conditions, and tables found in the context. Only if the context contains no relevant information at all, state that you do not have that information.

Context:
{context}

Question: {question}

Answer:"""
)

GUARDIAL_PROMPT = ChatPromptTemplate.from_template(
    """
You are a scope classifier for an HR assistant. Decide whether the question
below is something an HR assistant should answer (company leave policy,
reimbursement, code of conduct, or other internal HR topics) or something
out of scope (general knowledge, coding help, unrelated small talk, etc).
Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE.
question: {question}
"""
)

REFUSAL_MESSAGE = (
    "I'm an HR assistant and can only help with questions about company HR "
    "policies (leave, reimbursement, code of conduct, etc.). I don't have "
    "information to answer that question."
)


def foramt_docs(docs: List[Any]) -> str:
    """Original doc formatting function from notebook."""
    return ".\n\n".join(d.page_content for d in docs)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initializes the exact HuggingFace embedding model from notebook."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Initializes the exact RecursiveCharacterTextSplitter from notebook."""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )


def get_corpus_documents(corpus_path: str = CORPUS_PATH) -> List[Any]:
    """Loads PDF documents from the corpus directory using PyPDFDirectoryLoader."""
    loader = PyPDFDirectoryLoader(corpus_path)
    return loader.load()


def load_or_build_vectorstore(
    corpus_path: str = CORPUS_PATH,
    vectorstore_dir: str = VECTORSTORE_DIR,
    embedding_model: Optional[HuggingFaceEmbeddings] = None
) -> Tuple[FAISS, int]:
    """
    Loads persisted FAISS vectorstore if available; otherwise builds from corpus documents
    using the exact notebook pipeline and persists it for fast subsequent reloads.
    Returns (vectorstore, document_count).
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    index_file = os.path.join(vectorstore_dir, "index.faiss")
    pkl_file = os.path.join(vectorstore_dir, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        try:
            vectorstore = FAISS.load_local(
                vectorstore_dir,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            corpus_files = [f for f in os.listdir(corpus_path) if f.lower().endswith(".pdf")] if os.path.exists(corpus_path) else []
            return vectorstore, len(corpus_files)
        except Exception as e:
            print(f"Error loading local index, rebuilding: {e}")

    # Build from scratch
    documents = get_corpus_documents(corpus_path)
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    try:
        os.makedirs(vectorstore_dir, exist_ok=True)
        vectorstore.save_local(vectorstore_dir)
    except Exception as e:
        print(f"Could not persist vectorstore to disk: {e}")

    return vectorstore, len(documents)


def add_pdf_to_vectorstore(
    pdf_path: str,
    vectorstore: FAISS,
    vectorstore_dir: str = VECTORSTORE_DIR
) -> Tuple[int, int]:
    """
    Processes an uploaded PDF using the exact document loader and chunking logic,
    and appends it to the FAISS vectorstore.
    Returns (num_pages_loaded, num_chunks_created).
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)

    if chunks:
        vectorstore.add_documents(chunks)
        try:
            os.makedirs(vectorstore_dir, exist_ok=True)
            vectorstore.save_local(vectorstore_dir)
        except Exception as e:
            print(f"Warning: Could not save updated vectorstore: {e}")

    return len(documents), len(chunks)


def get_llm_model(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Any:
    """
    Initializes ChatGroq or ChatGoogleGenerativeAI.
    Defaults to Groq (openai/gpt-oss-120b) when Groq key is present.
    """
    groq_env_key = os.getenv("GROQ_API_KEY")
    gemini_env_key = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    selected_provider = provider or os.getenv("LLM_PROVIDER")
    if not selected_provider:
        if api_key and api_key.startswith("gsk_"):
            selected_provider = "groq"
        elif groq_env_key:
            selected_provider = "groq"
        else:
            selected_provider = "google-genai"

    if selected_provider.lower() == "groq":
        key = api_key or groq_env_key
        if not key:
            raise ValueError("Missing Groq API Key. Please provide GROQ_API_KEY in secrets, .env, or sidebar.")
        model = model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        return ChatGroq(
            model=model,
            groq_api_key=key,
            temperature=0.7,
            max_tokens=1500
        )
    else:
        key = api_key or gemini_env_key
        if not key:
            raise ValueError("Missing Google GenAI API Key. Please provide LLM_API_KEY.")
        model = model_name or DEFAULT_GEMINI_MODEL
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=0.7,
            max_tokens=1500
        )


def rag_chain(question: str, retriever: Any, llm_model: Any) -> Dict[str, Any]:
    """
    Exact RAG Chain:
    Retrieves top docs, formats context, invokes RAG_PROMPT with llm_model.
    """
    docs = retriever.invoke(question)
    context = foramt_docs(docs)
    chain = RAG_PROMPT | llm_model | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer, "sources": docs}


def ask_bot(question: str, retriever: Any, llm_model: Any) -> Dict[str, Any]:
    """
    Exact Guardrail + RAG implementation:
    Evaluates GUARDIAL_PROMPT to classify IN_SCOPE vs OUT_OF_SCOPE.
    Returns REFUSAL_MESSAGE if OUT_OF_SCOPE; otherwise invokes rag_chain.
    """
    guardial_chain = GUARDIAL_PROMPT | llm_model | StrOutputParser()
    verdict = guardial_chain.invoke({"question": question}).strip().upper()

    if "OUT_OF_SCOPE" in verdict:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "verdict": "OUT_OF_SCOPE"
        }

    result = rag_chain(question, retriever, llm_model)
    result["verdict"] = "IN_SCOPE"
    return result
