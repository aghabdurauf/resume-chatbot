import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

# Load environment variables
load_dotenv()

# Initialize Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configuration
RESUME_PDF_PATH = r"E:\Abdul Rauf\Resume\Optimized Resume\Agha Abdul Rauf Resume.pdf"
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

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = get_embeddings()

    # Create vector database
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name="resume_collection"
    )

    return vectordb

# Initialize vector database
try:
    vectordb = load_and_process_resume()
except Exception as e:
    st.error(f"Error loading resume: {str(e)}")
    vectordb = None

def retrieve_relevant_context(query, k=3):
    """Retrieve relevant chunks from vector database"""
    if vectordb is None:
        return ""

    # Retrieve similar documents
    docs = vectordb.similarity_search(query, k=k)

    # Combine retrieved chunks
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def generate_response(query, context):
    """Generate response using Gemini with retrieved context"""

    system_instruction = f"""You are a helpful assistant representing Abdul Rauf Agha.
    You have access to his resume and professional information.
    Answer questions about his experience, skills, projects, and qualifications based on the following context:

    CONTEXT:
    {context}

    Important guidelines:
    - Only provide information that is in the provided context
    - Be professional and concise
    - If asked about something not in the context, politely say you don't have that information
    - Highlight relevant achievements and metrics when appropriate
    - Be enthusiastic about Abdul Rauf's qualifications
    """

    # Initialize Gemini model
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction=system_instruction
    )

    # Generate response
    response = model.generate_content(query)
    return response.text

# Streamlit UI Configuration
st.set_page_config(
    page_title="Abdul Rauf's Resume Chatbot (RAG)",
    page_icon="💼",
    layout="wide"
)

# Title and description
st.title("💼 Abdul Rauf Agha - Resume Chatbot (RAG-Based)")
st.markdown("**Ask me anything about Abdul Rauf's professional experience, skills, and qualifications!**")
st.markdown("*This chatbot uses RAG (Retrieval Augmented Generation) technology*")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = "Hello! I'm Abdul Rauf's Resume Assistant powered by RAG. I retrieve relevant information from his resume to answer your questions accurately. What would you like to know?"
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
                context = retrieve_relevant_context(prompt, k=3)

                # Step 2: Generate response using Gemini with retrieved context
                assistant_response = generate_response(prompt, context)

                # Display response
                st.markdown(assistant_response)

                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                # Show retrieved context in expander (for debugging/transparency)
                with st.expander("🔍 View Retrieved Context (RAG)"):
                    st.text(context)

            except Exception as e:
                error_message = f"Error: {str(e)}\n\nPlease make sure your Gemini API key is set correctly in the .env file."
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
    - 🤖 Generates answers using Gemini AI

    **You can ask about:**
    - Work experience and projects
    - Technical skills and tools
    - Education and certifications
    - Specific achievements
    - Contact information

    **Example Questions:**
    - What AI projects has Abdul Rauf worked on?
    - What are his main technical skills?
    - Tell me about his experience with Power BI
    - What certifications does he have?
    - Has he worked with healthcare data?
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
        st.info("🧠 Model: all-MiniLM-L6-v2")
    else:
        st.error("❌ Vector DB Not Loaded")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
