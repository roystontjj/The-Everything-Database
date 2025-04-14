import streamlit as st
import google.generativeai as genai

# Page config and styling
st.set_page_config(page_title="Gemini AI Assistant", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .chat-container {
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .assistant-message {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("💬 Gemini AI Assistant")
st.markdown("Ask questions, get creative content, or request assistance with various tasks.")

# Get API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Display chat history
st.subheader("Conversation")
chat_container = st.container(height=400)
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><strong>You:</strong><br>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong><br>{message['content']}</div>", unsafe_allow_html=True)

# User input section
st.subheader("Your Question")
with st.form(key="prompt_form", clear_on_submit=True):
    user_prompt = st.text_area("Type your message here:", height=100, key="prompt_input")
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.form_submit_button("Send", use_container_width=True)
    with col2:
        if st.form_submit_button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat = model.start_chat(history=[])
            st.rerun()

# Process the user input when the form is submitted
if submit_button and user_prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Show a spinner while processing
    with st.spinner("Thinking..."):
        try:
            # Get response from Gemini
            response = st.session_state.chat.send_message(user_prompt)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
    
    # Rerun to update the UI with the new messages
    st.rerun()

# Add some footer information
st.markdown("---")
st.caption("Powered by Google Gemini 1.5 Flash")