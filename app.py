import time
import streamlit as st

# Import your back-end logic
from generations import (
    get_response,
    ingest_additional_knowledge,
    store_chat_in_pinecone,
    KNOWLEDGE_INDEX_NAME,
    CHAT_INDEX_NAME
)

def main():
    # ---------------------------------------------------
    # 1. Basic Page Config
    # ---------------------------------------------------
    st.set_page_config(
        page_title="Crustdata API Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # ---------------------------------------------------
    # 2. Custom CSS
    # ---------------------------------------------------
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fc;
    }
    .chat-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .user-message, .assistant-message {
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e1f5fe;
    }
    .assistant-message {
        background-color: #fff9c4;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🤖 Crustdata API Assistant")

    # Show which Pinecone indexes we're using
    st.write(f"**Knowledge Base Index:** `{KNOWLEDGE_INDEX_NAME}`")
    st.write(f"**Chat History Index:** `{CHAT_INDEX_NAME}`")

    # ---------------------------------------------------
    # 3. Initialize Session State for Chat
    # ---------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ---------------------------------------------------
    # 4. Knowledge Ingestion Section
    # ---------------------------------------------------
    st.subheader("Add New Knowledge")
    with st.expander("Ingest Additional Documentation or Q&A"):
        st.markdown(
            f"Paste **text** from Slack Q&As, user Q&As, or any new documentation below. "
            f"This data will be stored in **'{KNOWLEDGE_INDEX_NAME}'** for future queries."
        )
        doc_text = st.text_area("Text to Ingest", height=120, placeholder="Paste your doc or Q&A here...")
        doc_source = st.text_input("Source / Tag (optional)", placeholder="e.g., Slack #support channel")

        if st.button("Ingest Document"):
            if doc_text.strip():
                metadata = {"source": doc_source} if doc_source else {}
                success = ingest_additional_knowledge(doc_text, metadata)
                if success:
                    st.success("Document ingestion successful!")
                else:
                    st.error("Document ingestion failed.")
            else:
                st.warning("Please enter some text to ingest.")

    # ---------------------------------------------------
    # 5. Display Current Conversation
    # ---------------------------------------------------
    st.subheader("Chat with the Crustdata Assistant")
    st.write("Ask any question about Crustdata APIs, including newly ingested data.")

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # 6. Chat Input (Using a Form)
    # ---------------------------------------------------
    with st.form(key="chat_form"):
        user_query_local = st.text_input("Your question:", placeholder="E.g., How do I search by location?")
        submitted = st.form_submit_button("Submit Query")

        if submitted:
            # 1) Store user message
            if user_query_local.strip():
                st.session_state["messages"].append({"role": "user", "content": user_query_local})
                store_chat_in_pinecone("user", user_query_local)

                # 2) Generate assistant response
                with st.spinner("Thinking..."):
                    assistant_resp = get_response(user_query_local)
                    time.sleep(0.5)

                # 3) Store assistant response
                st.session_state["messages"].append({"role": "assistant", "content": assistant_resp})
                store_chat_in_pinecone("assistant", assistant_resp)

                st.success("Response received!")
            else:
                st.warning("Please enter a question before submitting.")

    # ---------------------------------------------------
    # 7. Clear Conversation Button
    # ---------------------------------------------------
    if st.button("Clear Conversation"):
        st.session_state["messages"] = []
        st.info("Conversation cleared. You can start fresh.")

    st.markdown("---")
    st.markdown("Powered by **OpenAI GPT** and **Pinecone** | © 2025 Crustdata")


if __name__ == "__main__":
    main()
