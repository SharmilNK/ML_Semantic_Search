import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import traceback
import os
import tempfile

# --- 1. Setup & Initialization ---

st.set_page_config(page_title="🤖 ML Class Semantic Search Engine")

st.title("🤖 ML Class Semantic Search Engine")
st.markdown("""
Search through your Machine Learning lecture slides using AI-powered semantic search.

**Features:**
- Understands meaning, not just keywords  
- Searches across all your ML class PowerPoints  
- Instant results with source attribution  
""")

# Initialize the Embedding Model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Connect to ChromaDB
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(tempfile.gettempdir(), "chromadb")

st.write(f"Connecting to ChromaDB at: `{db_path}`")

client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection("ml_class_slides")

# Example: add documents if empty
if collection.count() == 0:
    collection.add(
        documents=["Intro to ML", "Gradient Descent details"],
        metadatas=[{"source": "slides1", "slide": 1}, {"source": "slides2", "slide": 2}],
        ids=["doc1", "doc2"]
    )

# --- 2. Core Search Logic ---

def semantic_search(query, n_results=3):
    """Performs a semantic search against the ChromaDB collection."""
    try:
        query_embedding = model.encode([query])
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=int(n_results)
        )

        if not results['documents'] or not results['documents'][0]:
            return ["No relevant slides found."]

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        formatted_list = []
        for i, doc in enumerate(documents):
            source = metadatas[i].get('source', 'Unknown File')
            slide_num = metadatas[i].get('slide', 'Unknown Slide')
            result_str = (
                f"📄 **Source:** {source}\n"
                f"📊 **Slide:** {slide_num}\n"
                f"{'-' * 40}\n"
                f"{doc}\n"
                f"{'=' * 40}"
            )
            formatted_list.append(result_str)

        return formatted_list

    except Exception as e:
        return [f"Error during search: {str(e)}\n\n{traceback.format_exc()}"]

# --- 3. Streamlit UI ---

query = st.text_input("🔍 Search Your ML Class Notes", placeholder="e.g., 'How does gradient descent work?'")
num_results = st.slider("📊 Number of Results", min_value=1, max_value=5, value=3, step=1)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a search query!")
    else:
        results = semantic_search(query, n_results=num_results)
        st.markdown("### 🎯 Search Results")
        for r in results:
            st.markdown(r)

# --- 4. Examples Section ---
st.markdown("### Examples")
examples = [
    ("What is gradient descent?", 2),
    ("Explain backpropagation in neural networks", 3),
    ("How does regularization prevent overfitting?", 2)
]

for ex_query, ex_num in examples:
    if st.button(f"Try: {ex_query}"):
        results = semantic_search(ex_query, n_results=ex_num)
        st.markdown("### 🎯 Search Results")
        for r in results:
            st.markdown(r)
