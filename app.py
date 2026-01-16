import streamlit as st
from groq import Groq
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

# Initialize Groq client using Streamlit Secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Configuration - use relative path for deployment
RESUME_PDF_PATH = os.path.join(os.path.dirname(__file__), "resume.pdf")
CHROMA_DB_PATH = "./chroma_db"

# Initialize embeddings model
@st.cache_resource
def get_embeddings():
    """Initialize and cache the embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# Load and process PDF
@st.cache_resource
def load_and_process_resume():
    """Load PDF, extract text, chunk it, and create vector database"""

    # Extract text from PDF
    pdf_reader = PdfReader(RESUME_PDF_PATH)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks (larger chunks for better context)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = get_embeddings()

    # Delete existing collection if exists and create fresh
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        chroma_client.delete_collection("resume_collection")
    except:
        pass

    # Create vector database
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name="resume_collection"
    )

    return vectordb, chunks

# Initialize vector database
try:
    vectordb, resume_chunks = load_and_process_resume()
except Exception as e:
    st.error(f"Error loading resume: {str(e)}")
    vectordb = None
    resume_chunks = []

def retrieve_relevant_context(query, k=5):
    """Retrieve relevant chunks from vector database"""
    if vectordb is None:
        # Fallback to raw chunks if vectordb failed
        if resume_chunks:
            return "\n\n".join(resume_chunks[:k])
        return ""

    try:
        # Retrieve similar documents
        docs = vectordb.similarity_search(query, k=k)
        # Combine retrieved chunks
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        # Fallback to raw chunks
        if resume_chunks:
            return "\n\n".join(resume_chunks[:k])
        return ""

def generate_response(query, context):
    """Generate response using Groq Llama 3 with retrieved context"""

    system_prompt = """You are **Tipu**, a professional AI assistant representing **Abdul Rauf**.

Your sole responsibility is to answer questions about **Abdul Rauf's professional career** using **only the information retrieved from the knowledge base (RAG)**.

### Greeting Behavior

When a user greets with words like **Hello, Hi, Hey, Salaam**, or similar, respond exactly with:

> **Hey, happy to see you. I am Tipu, the assistant of Abdul Rauf. What would you like to know about Abdul Rauf's professional career?**

Do not add anything else.

### Answering Rules

* Always retrieve information from RAG before answering.
* Do **not** use prior knowledge, assumptions, or external information.
* Do **not** infer, guess, or fill gaps.
* If the answer is **not found in RAG**, respond with:

  > *I'm sorry, I don't have that information available in Abdul Rauf's professional records.*

### Tone & Style

* Be polite, professional, and concise.
* Stick strictly to the question asked.
* Do not add extra context, suggestions, or opinions.
* Do not exaggerate or rephrase facts beyond what exists in RAG.

### Hallucination Guardrail

* If RAG data is incomplete, unclear, or missing, clearly state the limitation.
* Never fabricate roles, skills, achievements, timelines, or experience.

Accuracy is mandatory. Creativity is not allowed.

### RAG CONTEXT (Use ONLY this information to answer):
{context}"""

    # Generate response using Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": query}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content

# Streamlit UI Configuration
st.set_page_config(
    page_title="Abdul Rauf's Resume Chatbot (RAG)",
    page_icon="💼",
    layout="wide"
)

# Title and description
st.title("💼 Tipu - Abdul Rauf's AI Assistant")
st.markdown("**Ask me anything about Abdul Rauf's professional career!**")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = "Hey, happy to see you. I am Tipu, the assistant of Abdul Rauf. What would you like to know about Abdul Rauf's professional career?"
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about Abdul Rauf's experience, skills, or projects..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant information..."):
            try:
                # Step 1: Retrieve relevant context from vector database
                context = retrieve_relevant_context(prompt, k=5)

                # Step 2: Generate response using Groq with retrieved context
                assistant_response = generate_response(prompt, context)

                # Display response
                st.markdown(assistant_response)

                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            except Exception as e:
                error_message = f"Error: {str(e)}\n\nPlease make sure your Groq API key is set correctly."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar with info
with st.sidebar:
    st.header("About This Chatbot")
    st.markdown("""
    This is a **RAG-based** AI chatbot that can answer questions about **Abdul Rauf Agha's** professional background.

    **RAG Technology:**
    - 📄 Loads resume from PDF
    - ✂️ Chunks text into smaller pieces
    - 🔢 Creates vector embeddings
    - 🗄️ Stores in ChromaDB vector database
    - 🔍 Retrieves only relevant chunks
    - 🤖 Generates answers using Llama 3

    **You can ask about:**
    - Work experience and projects
    - Technical skills and tools
    - Education and certifications
    - Specific achievements
    - Contact information
    """)

    st.divider()

    st.header("Contact Information")
    st.markdown("""
    **Email:** aghaabdulrauf@gmail.com
    **Phone:** +971558843078
    **Location:** Dubai, UAE
    """)

    st.divider()

    st.header("RAG Stats")
    if vectordb:
        st.success("✅ Vector DB Loaded")
        st.info("📊 Chunks: Multiple")
        st.info("🧠 Embeddings: all-MiniLM-L6-v2")
        st.info("🤖 LLM: Llama 3.1 8B Instant (Groq)")
    else:
        st.error("❌ Vector DB Not Loaded")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
