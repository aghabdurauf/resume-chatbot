import streamlit as st
from groq import Groq
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import numpy as np

# Page config
st.set_page_config(
    page_title="Abdul Rauf - Resume Chatbot (Hybrid)",
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
        st.error("⚠️ GROQ_API_KEY not found in secrets!")
        st.stop()
    return Groq(api_key=api_key)

# Initialize Supabase client
@st.cache_resource
def get_supabase_client():
    """Initialize Supabase client"""
    url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
    key = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))

    if not url or not key:
        st.error("⚠️ Supabase credentials not found!")
        st.markdown("""
        ### 🔧 Add to Streamlit Secrets:
        ```toml
        SUPABASE_URL = "your_supabase_url"
        SUPABASE_KEY = "your_supabase_anon_key"
        ```
        Get from: Supabase Dashboard → Settings → API
        """)
        st.stop()

    return create_client(url, key)

# Load tiny embedding model (20MB)
@st.cache_resource
def get_embeddings_model():
    """Load lightweight embedding model"""
    with st.spinner("Loading tiny AI model (20MB)..."):
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    return model

# Initialize vector store (one-time setup)
def initialize_vector_store(supabase: Client, model: SentenceTransformer):
    """Check if vectors exist, if not, create them"""

    # Check if table has data
    response = supabase.table('resume_embeddings').select('id').limit(1).execute()

    if len(response.data) > 0:
        st.success(f"✅ Vector store ready ({len(response.data)} embeddings)")
        return True

    # If no data, need to initialize
    st.warning("⚠️ Vector store is empty. Initializing...")

    try:
        # Load resume
        resume_path = os.path.join(os.path.dirname(__file__), "resume.pdf")
        pdf_reader = PdfReader(resume_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create chunks
        chunk_size = 500
        overlap = 50
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 50:
                chunks.append(chunk)

        st.info(f"📄 Created {len(chunks)} chunks")

        # Generate embeddings
        with st.spinner("Creating embeddings..."):
            embeddings = model.encode(chunks, show_progress_bar=False)

        # Upload to Supabase
        with st.spinner("Uploading to Supabase..."):
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                supabase.table('resume_embeddings').insert({
                    'content': chunk,
                    'embedding': embedding.tolist()
                }).execute()

        st.success(f"✅ Uploaded {len(chunks)} embeddings to Supabase!")
        return True

    except Exception as e:
        st.error(f"❌ Error initializing vector store: {e}")
        return False

# Search vectors in Supabase
def search_similar(query: str, supabase: Client, model: SentenceTransformer, k: int = 3):
    """Search for similar chunks in Supabase"""

    # Encode query
    query_embedding = model.encode([query])[0].tolist()

    # Search using RPC function (we'll create this)
    # For now, get all and search locally (not optimal but works)
    response = supabase.table('resume_embeddings').select('*').execute()

    if not response.data:
        return []

    # Calculate similarities
    similarities = []
    query_vec = np.array(query_embedding)

    for row in response.data:
        embedding = np.array(row['embedding'])
        # Cosine similarity
        similarity = np.dot(query_vec, embedding) / (
            np.linalg.norm(query_vec) * np.linalg.norm(embedding)
        )
        similarities.append((row['content'], similarity))

    # Sort and get top k
    similarities.sort(key=lambda x: x[1], reverse=True)

    return [content for content, _ in similarities[:k]]

# Generate response
def generate_response(query: str, context: str, client: Groq):
    """Generate response using Groq"""

    system_prompt = """You are **Tipu**, a professional AI assistant representing **Abdul Rauf**.

Your sole responsibility is to answer questions about **Abdul Rauf's professional career** using **only the information from the context below**.

### Greeting Behavior

When a user greets with words like **Hello, Hi, Hey, Salaam**, respond exactly with:

> **Hey, happy to see you. I am Tipu, the assistant of Abdul Rauf. What would you like to know about Abdul Rauf's professional career?**

### Answering Rules

* Use ONLY information from the context
* If answer not found, say: *I'm sorry, I don't have that information available in Abdul Rauf's professional records.*
* Be concise and professional

### CONTEXT:
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
    st.title("💼 Abdul Rauf - Resume Chatbot")
    st.markdown("**🚀 Hybrid RAG: Local Embeddings + Supabase Vector DB**")
    st.markdown("---")

    # Initialize clients
    groq_client = get_groq_client()
    supabase = get_supabase_client()
    model = get_embeddings_model()

    # Initialize vector store
    if 'initialized' not in st.session_state:
        st.session_state.initialized = initialize_vector_store(supabase, model)

    if not st.session_state.initialized:
        st.error("⚠️ Vector store initialization failed. Please check setup.")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        css_class = "user-message" if role == "user" else "assistant-message"
        label = "You" if role == "user" else "Tipu"

        st.markdown(f"""
        <div class="chat-message {css_class}">
            <b>{label}:</b><br>{content}
        </div>
        """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask me anything about Abdul Rauf's career...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        st.markdown(f"""
        <div class="chat-message user-message">
            <b>You:</b><br>{user_input}
        </div>
        """, unsafe_allow_html=True)

        # Search and generate
        with st.spinner("🔍 Searching Supabase..."):
            relevant_chunks = search_similar(user_input, supabase, model, k=3)
            context = "\n\n".join(relevant_chunks)

        with st.spinner("💭 Thinking..."):
            response = generate_response(user_input, context, groq_client)

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

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
        **Hybrid RAG Architecture:**
        - 🧠 Local: 20MB embedding model
        - 💾 Cloud: Supabase vector DB
        - ⚡ Fast & lightweight
        - 📊 ~200MB memory usage

        **Tech Stack:**
        - Groq AI (Llama 3.1)
        - Supabase pgvector
        - Sentence Transformers
        - Streamlit
        """)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("🌟 Optimized for Streamlit Cloud")

if __name__ == "__main__":
    main()
