import streamlit as st
from groq import Groq
import os
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(
    page_title="Abdul Rauf - Resume Chatbot (RAG)",
    page_icon="💼",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #1e2130;
        border-left: 4px solid #4a9eff;
    }
    .assistant-message {
        background-color: #1a1d29;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    """Initialize Groq client"""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found!")
        st.markdown("""
        ### 🔧 Add API key in Streamlit Secrets:
        ```toml
        GROQ_API_KEY = "your_key_here"
        ```
        Get key from: https://console.groq.com/keys
        """)
        st.stop()
    return Groq(api_key=api_key)

# Load tiny embedding model (only 20MB!)
@st.cache_resource
def get_embeddings_model():
    """Load lightweight embedding model - MUCH smaller than default"""
    with st.spinner("Loading tiny AI model (20MB)..."):
        # This model is 15x smaller than all-MiniLM-L6-v2 but still effective
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    return model

# Simple vector store using numpy (no ChromaDB needed!)
class SimpleVectorStore:
    """Lightweight vector store using just numpy"""
    def __init__(self):
        self.texts = []
        self.embeddings = []

    def add_texts(self, texts, embeddings):
        self.texts = texts
        self.embeddings = np.array(embeddings)

    def similarity_search(self, query_embedding, k=3):
        """Find top k similar texts using cosine similarity"""
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        similarities = similarities / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [self.texts[i] for i in top_k_indices]

# Load and process resume with RAG
@st.cache_resource
def load_resume_with_rag():
    """Load resume, chunk it, and create embeddings"""
    try:
        # Load PDF
        resume_path = os.path.join(os.path.dirname(__file__), "resume.pdf")
        pdf_reader = PdfReader(resume_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Simple chunking (no heavy libraries needed)
        chunk_size = 500  # Smaller chunks = less memory
        overlap = 50

        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 50:  # Skip tiny chunks
                chunks.append(chunk)

        st.info(f"📄 Created {len(chunks)} chunks from resume")

        # Get embeddings model
        model = get_embeddings_model()

        # Create embeddings (batched for memory efficiency)
        with st.spinner("Creating semantic search index..."):
            embeddings = model.encode(chunks, show_progress_bar=False)

        # Create simple vector store
        vector_store = SimpleVectorStore()
        vector_store.add_texts(chunks, embeddings)

        st.success("✅ RAG system ready!")

        return vector_store, model, text

    except Exception as e:
        st.error(f"❌ Error loading resume: {e}")
        st.stop()

# Retrieve relevant context
def retrieve_context(query, vector_store, model, k=3):
    """Get relevant chunks using semantic search"""
    # Encode query
    query_embedding = model.encode([query])[0]

    # Search
    relevant_chunks = vector_store.similarity_search(query_embedding, k=k)

    return "\n\n".join(relevant_chunks)

# Generate response
def generate_response(query, context, client):
    """Generate response using retrieved context"""

    system_prompt = """You are **Tipu**, a professional AI assistant representing **Abdul Rauf**.

Your sole responsibility is to answer questions about **Abdul Rauf's professional career** using **only the information from the context below**.

### Greeting Behavior

When a user greets with words like **Hello, Hi, Hey, Salaam**, respond exactly with:

> **Hey, happy to see you. I am Tipu, the assistant of Abdul Rauf. What would you like to know about Abdul Rauf's professional career?**

### Answering Rules

* Use ONLY information from the context below
* Do NOT use prior knowledge or assumptions
* If answer not found, say: *I'm sorry, I don't have that information available in Abdul Rauf's professional records.*
* Be concise and professional

### CONTEXT (Retrieved from Resume):
{context}"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": query}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main app
def main():
    # Header
    st.title("💼 Abdul Rauf - Resume Chatbot")
    st.markdown("**🚀 Powered by Optimized RAG + Groq AI**")
    st.markdown("---")

    # Initialize
    client = get_groq_client()
    vector_store, embeddings_model, full_text = load_resume_with_rag()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <b>You:</b><br>{content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <b>Tipu:</b><br>{content}
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask me anything about Abdul Rauf's career...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>You:</b><br>{user_input}
        </div>
        """, unsafe_allow_html=True)

        # Retrieve context and generate response
        with st.spinner("🔍 Searching resume..."):
            context = retrieve_context(user_input, vector_store, embeddings_model, k=3)

        with st.spinner("💭 Thinking..."):
            response = generate_response(user_input, context, client)

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display assistant message
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <b>Tipu:</b><br>{response}
        </div>
        """, unsafe_allow_html=True)

        st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Optimized RAG Features:**
        - 🔍 Semantic search
        - 💾 Tiny 20MB model
        - ⚡ Fast retrieval
        - 🎯 Accurate answers

        **Tech Stack:**
        - Groq AI (Llama 3.1)
        - Mini embeddings
        - Numpy vector store
        - Streamlit
        """)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("🌟 Optimized for Streamlit Cloud")

if __name__ == "__main__":
    main()
