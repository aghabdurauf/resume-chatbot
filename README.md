# Abdul Rauf's Resume Chatbot

An AI-powered chatbot that answers questions about Abdul Rauf Agha's professional experience, skills, and qualifications.

## Features

- Interactive chat interface powered by Google Gemini AI
- Answers questions about work experience, projects, skills, and certifications
- Professional and context-aware responses
- Clean and modern Streamlit UI
- FREE tier available with Gemini API!

## Setup Instructions

### 1. Get Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Select a project or create a new one
5. Copy the API key (save it somewhere safe!)

### 2. Configure the Application

1. The `.env` file has already been created for you in the project folder

2. Open `.env` file and add your Gemini API key:
   ```
   GEMINI_API_KEY=your-actual-api-key-here
   ```

### 3. Run the Application

1. Open Command Prompt or Terminal

2. Navigate to the project folder:
   ```
   cd "E:\Abdul Rauf\Claude_Test\Resume Chatbot"
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Your browser will automatically open to `http://localhost:8501`

## Usage

Once the app is running, you can ask questions like:

- "What AI projects has Abdul Rauf worked on?"
- "What are his main technical skills?"
- "Tell me about his experience with Power BI"
- "What certifications does he have?"
- "Has he worked with healthcare data?"
- "What is his current role?"

## Project Structure

```
Resume Chatbot/
├── app.py              # Main Streamlit application
├── .env                # Your API key (create this from .env.example)
├── .env.example        # Template for environment variables
└── README.md           # This file
```

## Technologies Used

- **Streamlit**: Web interface framework
- **Google Gemini AI**: AI language model (gemini-1.5-flash)
- **Python**: Programming language
- **python-dotenv**: Environment variable management

## Notes

- The chatbot only provides information from Abdul Rauf's resume
- Gemini API has a FREE tier with generous limits (perfect for learning!)
- Keep your API key secure and never share it publicly

## Contact

**Abdul Rauf Agha**
- Email: aghaabdulrauf@gmail.com
- Phone: +971558843078
- Location: Dubai, UAE
