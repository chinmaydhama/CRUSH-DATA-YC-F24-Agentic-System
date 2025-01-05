import streamlit as st
from generations import get_response
import time

def main():
    st.set_page_config(page_title="Crustdata API Assistant", page_icon="🤖", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .user-message {
        background-color: #e1f5fe;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #f0f4c3;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🤖 Crustdata API Assistant")

    # Sidebar with additional options
    st.sidebar.title("Options")
    show_context = st.sidebar.checkbox("Show Context", value=False)
    max_messages = st.sidebar.slider("Max Chat History", min_value=5, max_value=50, value=10)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages[-max_messages:]:
        with st.chat_message(message["role"]):
            st.markdown(f"<div class='{message['role']}-message'>{message['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # User input
    prompt = st.chat_input("What's your question about Crustdata API?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)

        # Show a loading spinner while generating response
        with st.spinner("Thinking..."):
            response = get_response(prompt)
            time.sleep(1)  # Simulate processing time

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)

        # Show context if enabled
        if show_context:
            with st.expander("View Context"):
                st.write("Context used for generating the response:")
                st.code(response[:200] + "...")  # Display first 200 characters of context

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown("Powered by OpenAI GPT-4 and Pinecone")

if __name__ == "__main__":
    main()
