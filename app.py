import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# UI Setup
st.set_page_config(page_title="General Chatbot", page_icon="🤖")
st.title("🤖 General AI Assistant")
st.write("Ask me anything! I am a general-purpose AI assistant powered by Gemini.")

# Initialize the QA chain
@st.cache_resource
def load_chat_model():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and friendly general-purpose AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm
    return chain

chat_chain = load_chat_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initialize Langchain message history
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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
