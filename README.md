# 🏢 Zyro Dynamics — Enterprise HR Intelligence RAG Assistant

An enterprise-grade, retrieval-augmented generation (RAG) assistant for **Zyro Dynamics** employees to query company policies, leave entitlements, health benefits, code of conduct, compensation bands, and IT security guidelines with grounded, verifiable answers and safety guardrails.

---

## 📌 Architectural Overview

This application preserves 100% of the original RAG pipeline and architecture developed in `code.ipynb`:

| Component | Specification |
| :--- | :--- |
| **LLM Provider** | Google GenAI (`google-genai` / `langchain-google-genai`) |
| **Foundation Model** | `gemini-3.7-flash` (temperature: `0.7`, max tokens: `500`) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Document Loaders** | `PyPDFDirectoryLoader` & `PyPDFLoader` (`pypdf`) |
| **Chunking Strategy** | `RecursiveCharacterTextSplitter` (chunk size: `800`, overlap: `100`) |
| **Vector Database** | `FAISS` (Facebook AI Similarity Search) |
| **Retrieval Strategy** | Dense similarity search (`k = 3`) |
| **Safety Guardrail** | Scope Classifier Prompt (enforces `IN_SCOPE` vs `OUT_OF_SCOPE` refusal) |

---

## 🗂️ Project Structure

```text
Intelligent RAG/
├── app.py                     # Streamlit application with custom SaaS dark theme
├── rag_pipeline.py            # Preserved RAG pipeline, retriever, LLM, and guardrails
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
- A Google AI Studio API Key ([Get an API Key](https://aistudio.google.com/))

### 2. Clone and Create Virtual Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd "Intelligent RAG"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
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

Edit `.env` and set your Google Gemini API key:
```env
LLM_API_KEY=AIzaSy...your_gemini_api_key_here
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

Follow these steps to deploy this application to **Streamlit Community Cloud**:

### 1. Push to GitHub
Ensure the project is committed to your GitHub repository:
```bash
git add .
git commit -m "Deploy: Zyro Dynamics HR RAG Assistant"
git push origin main
```

### 2. Connect Repository on Streamlit Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set the main file path to:
   ```text
   app.py
   ```

### 3. Configure Secrets on Streamlit Cloud
Under **Advanced settings** ➔ **Secrets**, paste your API key in TOML format:

```toml
LLM_API_KEY = "AIzaSy...your_gemini_api_key_here"
```
*(Alternatively, you can also provide `GOOGLE_API_KEY = "..."`)*

Click **Deploy**!

### 4. Handling Vector Database in Cloud
The default corpus (`zyro-dynamics-hr-corpus/`) contains 11 concise PDFs (~100 KB total). When deployed:
- If `vectorstore/faiss_index` is omitted or ignored by git, the application automatically builds the FAISS index from the PDFs in `zyro-dynamics-hr-corpus/` on the first startup (~10–15 seconds) and caches it in memory using Streamlit's `@st.cache_resource`.
- Subsequent user interactions and queries retrieve from memory in milliseconds.

---

## 🛡️ Guardrails & Evaluation

The pipeline includes an active guardrail classifier (`ask_bot`) that evaluates queries before retrieval:

1. **In-Scope Queries**:
   - Company policies, leave entitlements, health insurance, compensation, POSH, WFH rules, performance reviews.
   - Handled via `rag_chain` with `FAISS` retrieval (top-3 chunks) + Gemini 3.7 Flash generation.
   - Verified source citations (document name + page number + snippet) are rendered in the UI.

2. **Out-of-Scope Queries**:
   - General trivia, coding requests, competitor queries (e.g., *"What is Zoho's leave policy like?"*).
   - Flagged by the guardrail classifier and answered with the standard compliance refusal message:
     > *"I'm an HR assistant and can only help with questions about company HR policies (leave, reimbursement, code of conduct, etc.). I don't have information to answer that question."*

---

## 🔑 Environment Variables & Secrets Reference

| Variable | Scope | Description |
| :--- | :--- | :--- |
| `LLM_API_KEY` | `.env` / Streamlit Secrets | Primary Google Gemini API key used by `ChatGoogleGenerativeAI` |
| `GOOGLE_API_KEY` | `.env` / Streamlit Secrets | Standard fallback alias for Google GenAI |
| `custom_api_key` | Streamlit UI Sidebar | Allows on-the-fly override of the API key directly in the web UI |

---

## 📄 License
Internal proprietary tool for Zyro Dynamics HR Operations.

