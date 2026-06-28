import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- UI Setup & Custom CSS ---
st.set_page_config(
    page_title="Health Chatbot System", 
    page_icon="⚕️", 
    layout="centered",
    initial_sidebar_state="expanded"
)

import os
if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
    st.error("🔑 **API Key Missing!**\n\nIt looks like you are running this on Streamlit Cloud but haven't added your API key yet.\n\nTo fix this:\n1. Go to your Streamlit Cloud Dashboard.\n2. Click the three dots (⋮) next to your app and select **Settings**.\n3. Go to the **Secrets** tab.\n4. Paste this exact line and save:\n\n`GEMINI_API_KEY=\"your-actual-api-key\"`")
    st.stop()


st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Hide default Streamlit header/footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1A1C23;
        border-right: 1px solid #2D3748;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        border-radius: 20px;
        border: 1px solid #4A5568 !important;
        background-color: #2D3748 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8a/Google_Gemini_logo.svg", width=150)
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.markdown("---")
    st.markdown("### 💡 About")
    st.markdown("This is a premium health chatbot system powered by Google's latest **Gemini 2.5 Flash** model. It features a responsive UI and conversational memory.")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.langchain_history = []
        st.rerun()

# --- Main Content ---
st.markdown('<div class="main-title">⚕️ Health Chatbot System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your intelligent, AI-powered health and wellness companion.</div>', unsafe_allow_html=True)

# Initialize the QA chain
@st.cache_resource(hash_funcs={"_thread.RLock": lambda _: None})
def load_chat_model(temp):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly intelligent, empathetic, and helpful health and wellness chatbot. Provide general wellness advice, but always remind the user to consult a doctor for serious medical concerns. Always format your responses beautifully using markdown, lists, and bold text where appropriate."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm
    return chain

# Load model with current temperature
chat_chain = load_chat_model(temperature)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initialize Langchain message history
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# Display chat history with custom avatars
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "⚕️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a health question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="⚕️"):
        with st.spinner("Analyzing health information..."):
            try:
                # Pass history and input to the chain
                response = chat_chain.invoke({
                    "history": st.session_state.langchain_history,
                    "input": prompt
                })
                
                answer = response.content
                st.markdown(answer)
                
                # Update history states
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.langchain_history.append(HumanMessage(content=prompt))
                st.session_state.langchain_history.append(AIMessage(content=answer))
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
