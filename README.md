# 🏢 Zyro Dynamics — Enterprise HR Intelligence RAG Assistant

An enterprise-grade, retrieval-augmented generation (RAG) assistant for **Zyro Dynamics** employees to query company policies, leave entitlements, health benefits, code of conduct, compensation bands, and IT security guidelines with grounded, verifiable answers and safety guardrails.

---

## 📌 Architectural Overview

This application preserves the exact RAG pipeline, retrieval logic, prompts, and guardrails developed in `code.ipynb`, enhanced with high-speed inference via **Groq** and **Google GenAI**:

| Component | Specification |
| :--- | :--- |
| **LLM Inference** | **Groq** (Default: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`) & **Google GenAI** (`gemini-3.7-flash`) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Document Loaders** | `PyPDFDirectoryLoader` & `PyPDFLoader` (`pypdf`) |
| **Chunking Strategy** | `RecursiveCharacterTextSplitter` (chunk size: `800`, overlap: `100`) |
| **Vector Database** | `FAISS` (Facebook AI Similarity Search) with disk caching |
| **Retrieval Strategy** | Dense similarity search (`k = 3`) |
| **Safety Guardrail** | Scope Classifier Prompt (enforces `IN_SCOPE` vs `OUT_OF_SCOPE` refusal) |

---

## 🗂️ Project Structure

```text
Intelligent RAG/
├── app.py                     # Streamlit application with custom SaaS dark theme & provider switcher
├── rag_pipeline.py            # Preserved RAG pipeline, retriever, Groq/Gemini LLM, and guardrails
├── requirements.txt           # Production package dependencies
├── .env.example               # Template for required environment variables
├── .gitignore                 # Protects secrets, cache, and upload files
├── README.md                  # Complete documentation and deployment guide
├── code.ipynb                 # Original development notebook
├── test.csv                   # 20 benchmark test evaluation questions
├── zyro-dynamics-hr-corpus/   # 11 Official Zyro Dynamics HR Policy PDFs
│   ├── 00_Company_Profile.pdf
│   ├── 01_Employee_Handbook.pdf
│   ├── 02_Leave_Policy.pdf
│   ├── 03_Work_From_Home_Policy.pdf
│   ├── 04_Code_of_Conduct.pdf
│   ├── 05_Performance_Review_Policy.pdf
│   ├── 06_Compensation_and_Benefits_Policy.pdf
│   ├── 07_IT_and_Data_Security_Policy.pdf
│   ├── 08_Prevention_of_Sexual_Harassment_Policy.pdf
│   ├── 09_Onboarding_and_Separation_Policy.pdf
│   └── 10_Travel_and_Expense_Policy.pdf
└── vectorstore/
    └── faiss_index/           # Persisted FAISS index (index.faiss & index.pkl)
```

---

## 🚀 Local Development Setup

### 1. Prerequisites
- Python 3.10 to 3.14
- A Groq API Key ([Get a free Groq API Key](https://console.groq.com/)) or Google Gemini API Key

### 2. Clone and Create Virtual Environment
```bash
git clone <your-repo-url>
cd "Intelligent RAG"

python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

Configure your API key in `.env`:
```env
# Groq (Free, ultra-fast inference)
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b

# Or Google Gemini
LLM_API_KEY=AIzaSy...
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

### 1. Push to GitHub
```bash
git add .
git commit -m "Support Groq free inference model and update Streamlit UI"
git push origin main
```

### 2. Configure Secrets on Streamlit Cloud
Under your app's **Advanced settings** ➔ **Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_..."
GROQ_MODEL = "openai/gpt-oss-120b"

# Optional: Google Gemini fallback
LLM_API_KEY = "AIzaSy..."
```

Click **Save** or **Deploy**! Streamlit Community Cloud will automatically pull the latest commit and redeploy.

---

## 🛡️ Guardrails & Evaluation

The pipeline includes an active guardrail classifier (`ask_bot`) that evaluates queries before retrieval:

1. **In-Scope Queries**:
   - Company policies, leave entitlements, health insurance, compensation, POSH, WFH rules, performance reviews.
   - Handled via `rag_chain` with `FAISS` retrieval (top-3 chunks) + LLM generation.
   - Verified source citations (document name + page number + snippet) are rendered in the UI.

2. **Out-of-Scope Queries**:
   - General trivia, coding requests, competitor queries (e.g., *"What is Zoho's leave policy like?"*).
   - Flagged by the guardrail classifier and answered with the standard compliance refusal message:
     > *"I'm an HR assistant and can only help with questions about company HR policies (leave, reimbursement, code of conduct, etc.). I don't have information to answer that question."*

---

## 🔑 Environment Variables & Secrets Reference

| Variable | Scope | Description |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | `.env` / Streamlit Secrets | Groq API Key (starts with `gsk_`) |
| `GROQ_MODEL` | `.env` / Streamlit Secrets | Groq model id (default: `openai/gpt-oss-120b`) |
| `LLM_API_KEY` | `.env` / Streamlit Secrets | Google Gemini API key used by `ChatGoogleGenerativeAI` |
| `GOOGLE_API_KEY` | `.env` / Streamlit Secrets | Standard fallback alias for Google GenAI |
| `custom_groq_key` | Streamlit UI Sidebar | Allows on-the-fly entry of the Groq API key in the UI |

---

## 📄 License
Internal proprietary tool for Zyro Dynamics HR Operations.
