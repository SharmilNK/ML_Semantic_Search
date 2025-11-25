import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
import json
import traceback
import os
import sys

# --- 1. Setup & Initialization ---

# Initialize the Embedding Model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Connect to the existing ChromaDB
# Fix: Use absolute path to ensure we find the DB folder regardless of where the script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chromadb")

print(f"Connecting to ChromaDB at: {db_path}")

if not os.path.exists(db_path):
    print(f"⚠️ WARNING: The folder '{db_path}' was not found.")
    print("Make sure you have copied the 'chromadb' folder from your notebook directory to this folder.")

# We use the specific 0.4.x syntax which matches the pinned requirement
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection("ml_class_slides")

# --- 2. Core Search Logic ---

def semantic_search(query, n_results=3):
    """
    Performs a semantic search against the ChromaDB collection.
    """
    try:
        # Generate embedding for the query
        query_embedding = model.encode([query])
        
        # Query the collection
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=int(n_results)
        )
        
        formatted_list = []
        
        # Check if we have results
        if not results['documents'] or not results['documents'][0]:
            return ["No relevant slides found."]
            
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
            
        for i, doc in enumerate(documents):
            source = metadatas[i].get('source', 'Unknown File')
            slide_num = metadatas[i].get('slide', 'Unknown Slide')
            
            # Construct a readable string for this result
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
        # Return the error message so it's visible in the UI
        return [f"Error during search: {str(e)}"]

# --- 3. Interface Helper Functions ---

def format_results(results):
    """Format the output from semantic_search for display"""
    if results is None:
        return "No results found."
    if isinstance(results, list):
        return "\n\n".join([str(r) for r in results])
    return str(results)

def search_interface(query, num_results):
    """Wrapper for Gradio interface"""
    if not query.strip():
        return "Please enter a search query!"
    try:
        raw_results = semantic_search(query, n_results=int(num_results))
        return format_results(raw_results)
    except Exception as e:
        return f"Error: {e}\n\n{traceback.format_exc()}"

# --- 4. Build the Gradio App ---

title = "🤖 ML Class Semantic Search Engine"
description = """
Search through your Machine Learning lecture slides using AI-powered semantic search.

**Features:**
-  Understands meaning, not just keywords  
-  Searches across all your ML class PowerPoints  
-  Instant results with source attribution  
"""

examples = [
    ["What is gradient descent?", 2],
    ["Explain backpropagation in neural networks", 3],
    ["How does regularization prevent overfitting?", 2]
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="🔍 Search Your ML Class Notes",
                placeholder="e.g., 'How does gradient descent work?'",
                lines=2
            )
            num_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="📊 Number of Results")
            search_button = gr.Button("Search", variant="primary")
        with gr.Column():
            result_box = gr.Textbox(label="🎯 Search Results", lines=20, show_copy_button=True)

    search_button.click(fn=search_interface, inputs=[query_input, num_slider], outputs=result_box)
    gr.Examples(examples=examples, inputs=[query_input, num_slider])

if __name__ == "__main__":
    print("\n🚀 Launching Gradio app...")
    demo.launch()