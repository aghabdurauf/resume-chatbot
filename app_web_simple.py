from flask import Flask, render_template, request, jsonify
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configuration
RESUME_PDF_PATH = os.path.join(os.path.dirname(__file__), "resume.pdf")

# Initialize Groq client
groq_client = None
_resume_text = None
_initialized = False
_initialization_error = None


def get_groq_client():
    global groq_client
    if groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        groq_client = Groq(api_key=api_key)
    return groq_client


def load_resume():
    """Simple PDF loading without heavy ML dependencies"""
    global _resume_text
    from pypdf import PdfReader

    pdf_reader = PdfReader(RESUME_PDF_PATH)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    _resume_text = text
    return text


def ensure_initialized():
    """Initialize resume text on first request"""
    global _initialized, _initialization_error

    if _initialized:
        return True

    if _initialization_error:
        return False

    try:
        print("🔄 Loading resume...", flush=True)
        load_resume()
        _initialized = True
        print("✅ Resume loaded!", flush=True)
        return True
    except Exception as e:
        _initialization_error = str(e)
        print(f"❌ Initialization failed: {e}", flush=True)
        return False


def generate_response(query):
    """Generate response using full resume as context - simpler approach"""
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

    client = get_groq_client()
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt.format(resume=_resume_text)},
            {"role": "user", "content": query},
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "initialized": _initialized}), 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not ensure_initialized():
        return jsonify({
            "error": f"System initialization failed: {_initialization_error}"
        }), 503

    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        response = generate_response(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Startup
print("🚀 Flask app starting (simple version - no RAG)...", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
