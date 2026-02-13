import streamlit as st
from groq import Groq
import os
from pypdf import PdfReader

# Page config
st.set_page_config(
    page_title="Abdul Rauf - Resume Chatbot",
    page_icon="💼",
    layout="centered"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
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
    """Initialize Groq client with API key from Streamlit secrets"""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found in secrets or environment variables!")
        st.stop()
    return Groq(api_key=api_key)

# Load resume once
@st.cache_data
def load_resume():
    """Load and extract text from resume PDF"""
    try:
        resume_path = os.path.join(os.path.dirname(__file__), "resume.pdf")
        pdf_reader = PdfReader(resume_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"❌ Error loading resume: {e}")
        st.stop()

# Generate response
def generate_response(query, resume_text, client):
    """Generate response using Groq with full resume as context"""

    system_prompt = """You are **Tipu**, a professional AI assistant representing **Abdul Rauf**.

Your sole responsibility is to answer questions about **Abdul Rauf's professional career** using **only the information from the resume below**.

### Greeting Behavior

When a user greets with words like **Hello, Hi, Hey, Salaam**, or similar, respond exactly with:

> **Hey, happy to see you. I am Tipu, the assistant of Abdul Rauf. What would you like to know about Abdul Rauf's professional career?**

Do not add anything else.

### Answering Rules

* Use only information from the resume below
* Do **not** use prior knowledge, assumptions, or external information
* Do **not** infer, guess, or fill gaps
* If the answer is **not found in the resume**, respond with:

  > *I'm sorry, I don't have that information available in Abdul Rauf's professional records.*

### Tone & Style

* Be polite, professional, and concise
* Stick strictly to the question asked
* Do not add extra context, suggestions, or opinions

### RESUME:
{resume}"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt.format(resume=resume_text)},
                {"role": "user", "content": query}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# Main app
def main():
    # Header
    st.title("💼 Abdul Rauf - Resume Chatbot")
    st.markdown("**Powered by AI Assistant Tipu**")
    st.markdown("---")

    # Initialize
    client = get_groq_client()
    resume_text = load_resume()

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
    user_input = st.chat_input("Ask me anything about Abdul Rauf's professional career...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>You:</b><br>{user_input}
        </div>
        """, unsafe_allow_html=True)

        # Generate and display response
        with st.spinner("Thinking..."):
            response = generate_response(user_input, resume_text, client)

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display assistant message
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <b>Tipu:</b><br>{response}
        </div>
        """, unsafe_allow_html=True)

        # Rerun to update chat
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses AI to answer questions about **Abdul Rauf's** professional career.

        **Features:**
        - 💬 Natural conversation
        - 📄 Resume-based responses
        - ⚡ Fast & accurate

        **Powered by:**
        - 🤖 Groq AI (Llama 3.1)
        - 🎨 Streamlit
        """)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
