from flask import Flask, render_template, request, jsonify
from groq import Groq
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configuration
RESUME_PDF_PATH = os.path.join(os.path.dirname(__file__), "resume.pdf")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- RAG Setup ---

_embeddings = None
_vectordb = None
_resume_chunks = []
_initialized = False
_initialization_error = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


def load_and_process_resume():
    global _vectordb, _resume_chunks

    pdf_reader = PdfReader(RESUME_PDF_PATH)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    _resume_chunks = text_splitter.split_text(text)

    embeddings = get_embeddings()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        chroma_client.delete_collection("resume_collection")
    except Exception:
        pass

    _vectordb = Chroma.from_texts(
        texts=_resume_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name="resume_collection",
    )


def ensure_initialized():
    """Lazy initialization - only runs on first request"""
    global _initialized, _initialization_error

    if _initialized:
        return True

    if _initialization_error:
        return False

    try:
        print("🔄 Initializing RAG system (lazy loading)...")
        load_and_process_resume()
        _initialized = True
        print("✅ RAG system ready!")
        return True
    except Exception as e:
        _initialization_error = str(e)
        print(f"❌ Initialization failed: {e}")
        return False


def retrieve_relevant_context(query, k=5):
    if _vectordb is None:
        if _resume_chunks:
            return "\n\n".join(_resume_chunks[:k])
        return ""
    try:
        docs = _vectordb.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception:
        if _resume_chunks:
            return "\n\n".join(_resume_chunks[:k])
        return ""


def generate_response(query, context):
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

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": query},
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


# --- Flask Routes ---

@app.route("/health")
def health():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy", "initialized": _initialized}), 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    # Lazy initialization on first request
    if not ensure_initialized():
        return jsonify({
            "error": f"System initialization failed: {_initialization_error}. Please try again later."
        }), 503

    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        context = retrieve_relevant_context(user_message, k=5)
        response = generate_response(user_message, context)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Startup ---

print("🚀 Flask app starting... (RAG will initialize on first request)")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
