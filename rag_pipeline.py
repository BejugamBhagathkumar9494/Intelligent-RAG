"""
rag_pipeline.py - Core RAG and Guardrail Pipeline for Zyro Dynamics HR Assistant.

Designed to be simple, robust, and beginner-friendly for B.Tech AI students:
1. Loads all 11 HR policy PDFs using straightforward pypdf extraction.
2. Chunks documents using RecursiveCharacterTextSplitter (chunk_size=800, overlap=100).
3. Indexes chunks into FAISS with sentence-transformers/all-MiniLM-L6-v2 embeddings.
4. Auto-detects changes in the corpus using a fingerprint so outdated indexes are never loaded.
5. Filters out distant, irrelevant citations so answers and citations stay 100% accurate.
6. Evaluates safety guardrails (IN_SCOPE vs OUT_OF_SCOPE).
7. Synthesizes grounded answers using Groq (free, high-speed) or Google Gemini.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Document loading and processing
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings and Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain prompts and LLM interfaces
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load local environment variables from .env
load_dotenv(override=True)

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
CORPUS_PATH = "./zyro-dynamics-hr-corpus"
VECTORSTORE_DIR = "./vectorstore/faiss_index"

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
DEFAULT_TOP_K = 5

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Step 1: PDF Document Loading
# -----------------------------------------------------------------------------
def load_all_pdfs(corpus_path: str = CORPUS_PATH) -> List[Document]:
    """
    Loads all PDF documents from the given folder using pypdf.
    - Loops through every .pdf file in alphabetical order.
    - Extracts text page by page.
    - Attaches clean metadata: 'source' (filename only) and 'page' (0-indexed).
    - Gracefully skips blank or unreadable pages.
    """
    if not os.path.exists(corpus_path):
        print(f"Warning: Corpus directory '{corpus_path}' does not exist.")
        return []

    # Get all PDF filenames sorted alphabetically
    pdf_filenames = sorted([
        f for f in os.listdir(corpus_path)
        if f.lower().endswith(".pdf")
    ])

    loaded_documents = []

    for filename in pdf_filenames:
        file_path = os.path.join(corpus_path, filename)
        try:
            reader = PdfReader(file_path)
            for page_index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                # Only include pages that actually have readable text
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": filename,       # Clean filename (e.g. 04_Code_of_Conduct.pdf)
                            "page": page_index        # 0-indexed page number
                        }
                    )
                    loaded_documents.append(doc)
        except Exception as error:
            print(f"Error reading {filename}: {error}")

    print(f"Successfully loaded {len(loaded_documents)} pages from {len(pdf_filenames)} PDF files.")
    return loaded_documents


# Alias for backward compatibility
get_corpus_documents = load_all_pdfs


# -----------------------------------------------------------------------------
# Step 2: Document Chunking
# -----------------------------------------------------------------------------
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Creates the text splitter with chunk_size=800 and chunk_overlap=100."""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Splits loaded pages into chunks while preserving metadata."""
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document pages into {len(chunks)} text chunks.")
    return chunks


# -----------------------------------------------------------------------------
# Step 3: Embeddings
# -----------------------------------------------------------------------------
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Loads sentence-transformers/all-MiniLM-L6-v2 embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------------------------------------------------------
# Step 4: FAISS Vector Database Management with Corpus Fingerprinting
# -----------------------------------------------------------------------------
def get_corpus_fingerprint(corpus_path: str = CORPUS_PATH) -> str:
    """
    Computes a fingerprint of all PDF files in the corpus (filenames and file sizes).
    Used to verify if the saved FAISS index on disk matches the current PDFs.
    """
    if not os.path.exists(corpus_path):
        return ""
    
    pdf_files = sorted([f for f in os.listdir(corpus_path) if f.lower().endswith(".pdf")])
    fingerprint_entries = []
    for f in pdf_files:
        full_path = os.path.join(corpus_path, f)
        file_size = os.path.getsize(full_path)
        fingerprint_entries.append(f"{f}:{file_size}")
    
    return "|".join(fingerprint_entries)


def load_or_build_vectorstore(
    corpus_path: str = CORPUS_PATH,
    vectorstore_dir: str = VECTORSTORE_DIR,
    embedding_model: Optional[HuggingFaceEmbeddings] = None
) -> Tuple[FAISS, int]:
    """
    Loads persisted FAISS vectorstore if available and matches the corpus fingerprint.
    If the index is missing, outdated, or corrupted, it automatically rebuilds from
    the 11 PDF files and persists the updated index and fingerprint to disk.
    
    Returns:
        (vectorstore, number_of_pdf_files)
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    index_file = os.path.join(vectorstore_dir, "index.faiss")
    pkl_file = os.path.join(vectorstore_dir, "index.pkl")
    fingerprint_file = os.path.join(vectorstore_dir, "fingerprint.txt")

    current_fingerprint = get_corpus_fingerprint(corpus_path)
    corpus_pdf_count = len([f for f in os.listdir(corpus_path) if f.lower().endswith(".pdf")]) if os.path.exists(corpus_path) else 0

    # 1. Check if an existing index on disk is valid and up-to-date
    if os.path.exists(index_file) and os.path.exists(pkl_file) and os.path.exists(fingerprint_file):
        try:
            with open(fingerprint_file, "r", encoding="utf-8") as fp:
                saved_fingerprint = fp.read().strip()

            if saved_fingerprint == current_fingerprint and current_fingerprint != "":
                # The saved index perfectly matches the current PDFs
                vectorstore = FAISS.load_local(
                    vectorstore_dir,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                print("Loaded valid, up-to-date FAISS vectorstore from disk.")
                return vectorstore, corpus_pdf_count
            else:
                print("Corpus files have changed! Rebuilding FAISS vectorstore...")
        except Exception as load_err:
            print(f"Could not load existing index ({load_err}). Rebuilding from scratch...")

    # 2. Build index fresh from all PDFs in the corpus
    print("Building FAISS index from corpus documents...")
    documents = load_all_pdfs(corpus_path)
    chunks = chunk_documents(documents)
    
    if not chunks:
        raise ValueError(f"No documents found to index in '{corpus_path}'!")

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # 3. Save new index and fingerprint to disk
    try:
        os.makedirs(vectorstore_dir, exist_ok=True)
        vectorstore.save_local(vectorstore_dir)
        with open(fingerprint_file, "w", encoding="utf-8") as fp:
            fp.write(current_fingerprint)
        print("Persisted rebuilt FAISS index and corpus fingerprint to disk.")
    except Exception as save_err:
        print(f"Warning: Could not save vectorstore to disk: {save_err}")

    return vectorstore, corpus_pdf_count


def add_pdf_to_vectorstore(
    pdf_path: str,
    vectorstore: FAISS,
    vectorstore_dir: str = VECTORSTORE_DIR
) -> Tuple[int, int]:
    """
    Extracts text from an uploaded PDF, chunks it, and appends to the vector database.
    Returns (pages_loaded, chunks_created).
    """
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    docs = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": page_idx}
            ))

    if not docs:
        return 0, 0

    chunks = chunk_documents(docs)
    vectorstore.add_documents(chunks)

    # Save updated vectorstore
    try:
        os.makedirs(vectorstore_dir, exist_ok=True)
        vectorstore.save_local(vectorstore_dir)
    except Exception as e:
        print(f"Warning: Could not save updated vectorstore: {e}")

    return len(docs), len(chunks)


# -----------------------------------------------------------------------------
# Step 5: LLM Model Initialization
# -----------------------------------------------------------------------------
def get_llm_model(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Any:
    """
    Initializes ChatGroq or ChatGoogleGenerativeAI with robust key resolution.
    Defaults to Groq (openai/gpt-oss-120b) for fast, free inference.
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
            raise ValueError("Missing Groq API Key. Please provide GROQ_API_KEY.")
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


# -----------------------------------------------------------------------------
# Step 6: Citation Filtering & RAG Chain
# -----------------------------------------------------------------------------
def filter_relevant_chunks(
    scored_docs: List[Tuple[Document, float]],
    max_score_delta: float = 0.20
) -> List[Document]:
    """
    Filters out chunks that are significantly less relevant than the top match.
    This eliminates irrelevant citations (e.g. POSH citations when asking about Code of Conduct).
    
    Logic:
    1. The first document is the most relevant (has the lowest L2 distance).
    2. Any chunk from the exact same document is kept.
    3. Any chunk from other documents is only kept if its score is very close
       to the best match (within max_score_delta) and overall distance is acceptable.
    """
    if not scored_docs:
        return []

    best_doc, best_score = scored_docs[0]
    best_source = best_doc.metadata.get("source", "")

    filtered_docs = []
    for doc, score in scored_docs:
        doc_source = doc.metadata.get("source", "")
        
        # Keep if from the same document as top match
        if doc_source == best_source:
            filtered_docs.append(doc)
        # Keep other documents only if score is closely related
        elif (score - best_score) <= max_score_delta and score <= 1.15:
            filtered_docs.append(doc)

    # Fallback to top match if filter emptied the list
    if not filtered_docs:
        filtered_docs = [best_doc]

    return filtered_docs


def format_docs(docs: List[Document]) -> str:
    """Combines retrieved document contents into a single context string."""
    return "\n\n".join(d.page_content for d in docs)


# Alias for backward compatibility
foramt_docs = format_docs


def rag_chain(
    question: str,
    vectorstore_or_retriever: Any = None,
    llm_model: Any = None,
    top_k: int = DEFAULT_TOP_K,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Executes the RAG pipeline:
    1. Retrieves top_k chunks with similarity scores.
    2. Filters out irrelevant citations using score-gap filtering.
    3. Formats context.
    4. Generates comprehensive grounded response with the LLM.
    """
    # 1. Determine top_k (from kwargs, retriever search_kwargs, or parameter)
    k = top_k
    if "top_k" in kwargs:
        k = kwargs["top_k"]
    elif hasattr(vectorstore_or_retriever, "search_kwargs") and isinstance(vectorstore_or_retriever.search_kwargs, dict):
        if "k" in vectorstore_or_retriever.search_kwargs:
            k = vectorstore_or_retriever.search_kwargs["k"]

    # 2. Resolve vectorstore
    if vectorstore_or_retriever is None:
        vs, _ = load_or_build_vectorstore()
    elif hasattr(vectorstore_or_retriever, "similarity_search_with_score"):
        vs = vectorstore_or_retriever
    elif hasattr(vectorstore_or_retriever, "vectorstore"):
        vs = vectorstore_or_retriever.vectorstore
    else:
        vs = vectorstore_or_retriever

    # 3. Resolve LLM model
    if llm_model is None:
        llm_model = get_llm_model()

    # 4. Similarity search with score
    scored_results = vs.similarity_search_with_score(question, k=k)

    # 5. Filter irrelevant chunks
    filtered_docs = filter_relevant_chunks(scored_results, max_score_delta=0.20)

    # 6. Format context string
    context = format_docs(filtered_docs)

    # 7. Generate answer with LLM
    chain = RAG_PROMPT | llm_model | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": filtered_docs
    }


# -----------------------------------------------------------------------------
# Step 7: Guardrail Scope Classification & ask_bot
# -----------------------------------------------------------------------------
def ask_bot(
    question: str,
    vectorstore_or_retriever: Any = None,
    llm_model: Any = None,
    top_k: int = DEFAULT_TOP_K,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluates guardrail prompt first:
    - If question is OUT_OF_SCOPE, immediately returns REFUSAL_MESSAGE.
    - If IN_SCOPE, executes the RAG pipeline.

    Accepts any calling convention:
    - ask_bot(question)
    - ask_bot(question, retriever, llm)
    - ask_bot(question, vectorstore, llm)
    - ask_bot(question, retriever, llm, top_k=5)
    """
    k = top_k
    if "top_k" in kwargs:
        k = kwargs["top_k"]

    if llm_model is None:
        llm_model = get_llm_model()

    guardrail_chain = GUARDIAL_PROMPT | llm_model | StrOutputParser()
    verdict = guardrail_chain.invoke({"question": question}).strip().upper()

    if "OUT_OF_SCOPE" in verdict:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "verdict": "OUT_OF_SCOPE"
        }

    result = rag_chain(question, vectorstore_or_retriever, llm_model, top_k=k, *args, **kwargs)
    result["verdict"] = "IN_SCOPE"
    return result
