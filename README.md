# ML_Semantic_Search

# 🤖 ML Class Semantic Search Engine

A **semantic search app** that lets you explore your local PowerPoint lecture slides using **AI embeddings** and **ChromaDB** — all powered by **Gradio** for a clean web UI.

---

## 🚀 Features

-  **Semantic search** — understands meaning, not just keywords  
-  **Embeddings with Sentence Transformers** (`all-MiniLM-L6-v2`)  
-  **Automatic slide extraction** from `.pptx` files  
-  **Persistent ChromaDB** database (stored locally under `./chroma_db/`)  
-  **Gradio web app** interface for ingestion, persistence, and searching  
-  Simple deployment to **Hugging Face Spaces**

---

##  Project Structure
├── app.py # Main Gradio application
├── requirements.txt # Dependencies
├── chroma_db/ # Auto-created local Chroma database
└── README.md # Project documentation

##  Installation

1. **Clone or copy** this repository.
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate     # on macOS/Linux
   venv\Scripts\activate        # on Windows
Install dependencies:
pip install -r requirements.txt

## Run the App : https://mlsemanticsearch-mt8yznf5r6shz4yzn9sb2q.streamlit.app/







