# 🧭 AIVenture – AI Travel Assistant

An intelligent **RAG-based (Retrieval-Augmented Generation) Travel Assistant** that helps users explore destinations, generate itineraries, and extract insights from travel guides using AI.

---

## 🚀 Overview

AIVenture allows users to:

* 📄 Upload travel guide PDFs
* 🔍 Ask questions about destinations
* 🧠 Get AI-powered, context-aware answers
* 🗺️ Generate travel plans and recommendations

The system combines **vector search (Qdrant)** with **LLM-based generation (FLAN-T5)** to deliver accurate and grounded responses.

---

## 🧠 Architecture

```text
User → Streamlit UI → FastAPI Backend
                           ↓
                    Qdrant Vector DB
                           ↓
               HuggingFace (FLAN-T5 Model)
```

---

## ⚙️ Tech Stack

### 🔹 Frontend

* Streamlit

### 🔹 Backend

* FastAPI
* Uvicorn

### 🔹 AI / ML

* HuggingFace Transformers (FLAN-T5-Base)
* Sentence Transformers (MiniLM)
* LangChain

### 🔹 Vector Database

* Qdrant

### 🔹 Others

* PyPDF (PDF parsing)
* Python-dotenv

---

## 📂 Project Structure

```bash
AIVenture/
│── src/
│   ├── main.py            # FastAPI app
│   ├── generator.py       # LLM response generation
│   ├── retriever.py       # Vector search logic
│   ├── ingest.py          # PDF ingestion pipeline
│   ├── embeddings.py      # Embedding model
│   ├── vectorstores.py    # Qdrant setup
│   ├── config.py          # Config variables
│
│── app.py                 # Streamlit frontend
│── requirements.txt
│── README.md
```

---

## ✨ Features

* 🔎 **Semantic Search** using vector embeddings
* 📄 **PDF Knowledge Ingestion**
* 🤖 **LLM-based Answer Generation**
* 🧭 **Travel Itinerary Suggestions**
* ⚡ **FastAPI + Streamlit Integration**
* ☁️ **Deployable on Render + Streamlit Cloud**

---

## ⚡ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Jayanesh2494/AIVenture-Travel-Assistant.git
cd AIVenture-Travel-Assistant
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Setup Environment Variables

Create a `.env` file:

```env
QDRANT_HOST=your_qdrant_url
QDRANT_API_KEY=your_api_key
MODEL_PROVIDER=huggingface
```

---

## ▶️ Run the Project

### Start Backend (FastAPI)

```bash
uvicorn src.main:app --reload
```

---

### Start Frontend (Streamlit)

```bash
streamlit run app.py
```

---



## 📌 Key Design Decisions

* Used **RAG architecture** to avoid hallucinations
* Chose **MiniLM embeddings** for speed and efficiency
* Optimized ingestion using:

  * batch upserts
  * reduced chunk size
* Used **FLAN-T5-Base** for lightweight deployment

---

## ⚠️ Limitations

* Free-tier deployment may cause:

  * cold starts
  * slower inference
* Large PDFs may increase processing time
* Not optimized for high concurrency

---

## 🙌 Acknowledgements

* HuggingFace Transformers
* LangChain
* Qdrant
* Streamlit
* FastAPI

---

## 📬 Contact

**Author:** Jayanesh D
🔗 GitHub: https://github.com/Jayanesh2494

---

⭐ If you like this project, consider giving it a star!
