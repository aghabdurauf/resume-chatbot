import streamlit as st
from openai import OpenAI
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

# Initialize GLM client
@st.cache_resource
def get_glm_client():
    """Initialize GLM (ZhipuAI) client"""
    api_key = st.secrets.get("GLM_API_KEY", os.environ.get("GLM_API_KEY"))
    if not api_key:
        st.error("⚠️ GLM_API_KEY not found in secrets!")
        st.stop()
    return OpenAI(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

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

    # Check if table has data and get actual count
    response = supabase.table('resume_embeddings').select('id', count='exact').execute()

    if len(response.data) > 0:
        total_count = response.count if hasattr(response, 'count') else len(response.data)
        st.success(f"✅ Vector store ready ({total_count} embeddings)")
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
    query_embedding = model.encode([query])[0]

    # Search using RPC function (we'll create this)
    # For now, get all and search locally (not optimal but works)
    response = supabase.table('resume_embeddings').select('*').execute()

    if not response.data:
        st.warning(f"⚠️ No embeddings found in Supabase!")
        return ["No resume data found. Please check if embeddings were uploaded."]

    st.info(f"📊 Database: Found {len(response.data)} embeddings")

    # Calculate similarities
    similarities = []
    query_vec = np.array(query_embedding, dtype=np.float32)

    for row in response.data:
        try:
            # Convert embedding to numpy array with explicit dtype
            if row.get('embedding') is None:
                continue  # Skip rows with null embeddings

            # Handle both string and list formats
            embedding_data = row['embedding']
            if isinstance(embedding_data, str):
                import json
                embedding_data = json.loads(embedding_data)

            embedding = np.array(embedding_data, dtype=np.float32)

            # Validate embedding shape
            if embedding.shape != query_vec.shape:
                continue  # Skip mismatched dimensions

            # Cosine similarity with safe division
            query_norm = np.linalg.norm(query_vec)
            embedding_norm = np.linalg.norm(embedding)

            if query_norm > 0 and embedding_norm > 0:
                similarity = np.dot(query_vec, embedding) / (query_norm * embedding_norm)
            else:
                similarity = 0.0

            similarities.append((row['content'], float(similarity)))

        except Exception as e:
            # Skip problematic embeddings
            print(f"Skipping row due to error: {e}")
            continue

    # Check if we found any valid similarities
    if not similarities:
        return [], []  # Return empty lists for content and scores

    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Filter by minimum similarity threshold (0.3 = 30% similar)
    SIMILARITY_THRESHOLD = 0.3
    filtered = [(content, score) for content, score in similarities[:k] if score >= SIMILARITY_THRESHOLD]

    if not filtered:
        st.warning(f"⚠️ No chunks met similarity threshold ({SIMILARITY_THRESHOLD}). Best score: {similarities[0][1]:.3f}")
        return [], []

    # Return separate lists of content and scores
    contents = [content for content, _ in filtered]
    scores = [score for _, score in filtered]

    return contents, scores

# Generate response
def generate_response(query: str, context: str, client: OpenAI, conversation_history: list = None):
    """Generate response using Groq with conversation history"""

    system_prompt = """You are **Tipu**, a professional AI assistant representing **Abdul Rauf**.

Your sole responsibility is to answer questions about **Abdul Rauf's professional career** using **only the information from the context below**.

### Greeting Behavior

When a user greets with words like **Hello, Hi, Hey, Salaam**, respond exactly with:

> **Hey, happy to see you. I am Tipu, the assistant of Abdul Rauf. What would you like to know about Abdul Rauf's professional career?**

### CRITICAL RULES - NO EXCEPTIONS:

1. **NEVER make up or infer information** - if it's not explicitly in the context, say you don't have it
2. **NEVER mix up companies** - each company mentioned in context is different
3. **Quote exact company names** from the context - don't paraphrase or guess
4. **Verify dates and details** match the exact company being asked about
5. **If uncertain, say you don't know** - accuracy is more important than answering

### When answering:
* First, find the exact company/topic mentioned in the question within the context
* Second, extract ONLY the information related to that specific company/topic
* Third, respond using ONLY what you found - no additions or assumptions
* If the specific information isn't in the context, say: *I'm sorry, I don't have that information available in Abdul Rauf's professional records.*

### CONTEXT (your ONLY source of truth):
{context}"""

    try:
        # Build messages with conversation history
        messages = [
            {"role": "system", "content": system_prompt.format(context=context)}
        ]

        # Add conversation history (last 6 messages to keep context manageable)
        if conversation_history:
            messages.extend(conversation_history[-6:])

        # Add current query
        messages.append({"role": "user", "content": query})

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="glm-4-flash",
            temperature=0.5,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main app
def main():
    st.title("💼 Abdul Rauf - Resume Chatbot")
    st.markdown("**🚀 Hybrid RAG: Local Embeddings + Supabase Vector DB | Powered by GLM-4-Flash**")
    st.markdown("---")

    # Initialize clients
    groq_client = get_glm_client()
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
            relevant_chunks, similarity_scores = search_similar(user_input, supabase, model, k=5)

            if not relevant_chunks:
                st.error("❌ No relevant information found. Please try rephrasing your question.")
                st.stop()

            context = "\n\n".join(relevant_chunks)

        # DEBUG: Show what was found with similarity scores
        with st.expander("🔍 Debug: Retrieved Context", expanded=False):
            st.write(f"**Found {len(relevant_chunks)} relevant chunks**")
            st.write("**Similarity Scores (higher = more relevant):**")
            for i, (chunk, score) in enumerate(zip(relevant_chunks, similarity_scores), 1):
                st.write(f"**Chunk {i} - Score: {score:.3f}** ({score*100:.1f}% match)")
                st.text(f"{chunk[:200]}...")
                st.divider()
            st.text(f"Full context length: {len(context)} characters")

        with st.spinner("💭 Thinking..."):
            # Pass conversation history for context-aware responses
            response = generate_response(
                user_input,
                context,
                groq_client,
                conversation_history=st.session_state.messages[:-1]  # Exclude current message
            )

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
        - GLM-4-Flash (ZhipuAI)
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
