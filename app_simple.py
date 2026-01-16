import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Resume content as context
RESUME_CONTEXT = """
Abdul Rauf Agha
Business Intelligence / Data Engineer / AI Automation Developer
Location: Dubai, UAE
Phone: +971558843078
Email: aghaabdulrauf@gmail.com
LinkedIn: LinkedIn Profile
GitHub: GitHub Profile

PROFESSIONAL SUMMARY:
- Engineered and optimized large-scale Data warehouse and ETL processes
- Spearhead end-to-end Big Data and data warehousing initiatives in Agile settings
- Delivered advanced data processing solutions for anomaly detection and operational optimization
- Specialized in data architecture and schema design

CURRENT POSITION:
Senior Consultant (Data Product Developer) at SMART SOLUTIONS LLC, Dubai, UAE (Jul 2024 - Present)
- Led multiple client projects using Agile methodology
- Developed Business Automation and AI Agents including:
  * Procure to Pay AI Automation
  * Motor Claims Surveyor AI Agent
  * Loan Underwriting Approval AI Agent
  * Complaint Automation with Sentimental Analysis
  * Contract Management AI Agent
  * Data Extraction and Entry AI Agents

KEY PROJECTS:
1. Ministry of Health Sultanate of Oman:
   - Big Data platform deployment for healthcare
   - Developed Spark applications and ETL/ELT pipelines
   - Improved data fetch speed by 30%
   - Reduced manual data retrieval time by 50%

2. National Finance Company, Oman:
   - ETL development for retail finance
   - Designed data marts for Credit Risk, Treasury, and Loans
   - Led Power BI migration

3. Abu Dhabi National Takaful:
   - Built ETL pipelines in SSIS
   - Integrated RPA bots with Premia platform
   - Reduced manual processing by 60%

4. KTC International, Dubai:
   - Real-time parking analytics platform
   - Improved parking utilization by 30%
   - Reduced congestion through IoT integration

5. Al-Jazeera Takaful, KSA:
   - SSIS ETL pipelines for motor insurance
   - Reduced query times by 50%
   - Boosted user interaction by 60%

PREVIOUS EXPERIENCE:
Data Engineer / BI Consultant at BLUTECH CONSULTING (Feb 2022 - Jun 2024)
- Worked with Habib Bank Limited and Bank Alfalah
- Improved data accuracy by 30%
- Enhanced fraud detection capabilities

Business Intelligence Developer at BADRI MANAGEMENT CONSULTANCY (Jan 2020 - Jan 2022)
- Data modeling for Motor Insurance
- Developed dashboards for Walaa Insurance (Saudi Arabia)

EDUCATION:
Bachelor of Science in Computer Science
Karachi Institute of Economics & Technology (KIET), Dec 2017

CERTIFICATIONS:
- Microsoft Power BI Data Analyst (PL-300)
- Microsoft Azure Data Fundamentals (DP-900)
- Alteryx Foundational Micro-Credentials
- IBM DataOps Methodology

TECHNICAL SKILLS:
- Dashboard & Reporting: Power BI, Qlik Sense, Tableau, QlikView
- Databases: MS SQL Server, Oracle SQL, PostgreSQL, MySQL, MongoDB, Pinecone, Supabase
- ETL Tools: Azure Data Factory, SSIS, DBT, Alteryx, KNIME
- Programming: Python, PySpark
- Big Data: Hadoop, Hive, Impala, Azure Data Lake, Azure Synapse Analytics
- Cloud: Microsoft Azure, Microsoft Fabric
- Automation: Power Automate, n8n, Zapier, Flowise
- GenAI: OpenAI, Ollama, LangChain, LlamaParse
"""

# Streamlit UI Configuration
st.set_page_config(
    page_title="Abdul Rauf's Resume Chatbot",
    page_icon="💼",
    layout="wide"
)

# Title and description
st.title("💼 Abdul Rauf Agha - Resume Chatbot")
st.markdown("**Ask me anything about Abdul Rauf's professional experience, skills, and qualifications!**")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = "Hello! I'm Abdul Rauf's Resume Assistant. Feel free to ask me about his experience, skills, projects, or qualifications. What would you like to know?"
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
        with st.spinner("Thinking..."):
            try:
                # Create system instruction with resume context
                system_instruction = f"""You are a helpful assistant representing Abdul Rauf Agha.
                You have access to his complete resume and professional information.
                Answer questions about his experience, skills, projects, and qualifications based on the following resume:

                {RESUME_CONTEXT}

                Important guidelines:
                - Only provide information that is in the resume
                - Be professional and concise
                - If asked about something not in the resume, politely say you don't have that information
                - Highlight relevant achievements and metrics when appropriate
                - Be enthusiastic about Abdul Rauf's qualifications
                """

                # Initialize Gemini model
                model = genai.GenerativeModel(
                    model_name='gemini-1.5-flash',
                    system_instruction=system_instruction
                )

                # Build conversation history for context
                chat_history = []
                for msg in st.session_state.messages[-10:]:
                    if msg["role"] == "user":
                        chat_history.append({"role": "user", "parts": [msg["content"]]})
                    elif msg["role"] == "assistant":
                        chat_history.append({"role": "model", "parts": [msg["content"]]})

                # Start chat with history
                chat = model.start_chat(history=chat_history[:-1])  # Exclude the current user message

                # Send message and get response
                response = chat.send_message(prompt)
                assistant_response = response.text

                # Display response
                st.markdown(assistant_response)

                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            except Exception as e:
                error_message = f"Error: {str(e)}\n\nPlease make sure your Gemini API key is set correctly in the .env file."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar with info
with st.sidebar:
    st.header("About This Chatbot")
    st.markdown("""
    This is an AI-powered chatbot that can answer questions about **Abdul Rauf Agha's** professional background.

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

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
