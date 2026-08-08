import streamlit as st
import os
import sys
import warnings

@st.cache_resource(show_spinner=False)
def load_agent():
    from framework_pipeline.langchain_agent import app as agent_app
    return agent_app

warnings.filterwarnings("ignore")

st.set_page_config(page_title="EU AI Act Assistant", page_icon="🇪🇺", layout="wide")

st.markdown("""
<style>
    /* Main background */
    [data-testid="stAppViewContainer"] {
        background-color: #121418;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1c21;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    /* Make header transparent */
    [data-testid="stHeader"] {
        background: transparent;
    }
    /* Title styling */
    h1 {
        font-weight: 800;
        font-size: 3.5rem !important;
        color: #D3E07C;
        margin-bottom: 0rem;
    }
    /* Subtitle styling */
    .subtitle {
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background-color: #1a1c21;
        color: #FFFFFF;
    }
    .stChatMessage.user {
        background-color: #23272e;
        border-left: 4px solid #4A90E2;
    }
    .stChatMessage.assistant {
        background-color: #1a1c21;
        border-left: 4px solid #D3E07C;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #D3E07C;
    }
    [data-testid="stExpanderDetails"] [data-testid="stAlert"] {
        background-color: rgba(30, 37, 50, 0.4) !important;
        padding: 12px !important;
        border-radius: 8px !important;
        border-left: 3px solid #4A90E2 !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stExpanderDetails"] [data-testid="stAlert"] p, [data-testid="stExpanderDetails"] [data-testid="stAlert"] li {
        font-size: 0.85rem !important;
    }
    /* Status styling */
    .stStatus {
        border-radius: 10px;
        background-color: #1a1c21;
        border: 1px solid #4A90E2;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>🇪🇺 EU AI Act Compliance Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Navigate the complexities of the EU AI Act with confidence. This intelligent assistant cross-references official documentation to provide accurate, citation-backed answers you can trust.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.sidebar.markdown("### 🌐 Connect with Me")

    st.sidebar.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .social-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 10px;
        }
        .social-link {
            display: flex;
            align-items: center;
            gap: 10px;
            text-decoration: none !important;
            font-weight: 500;
            font-size: 15px;
        }
        .social-link:hover {
            opacity: 0.8;
        }
        .fa-linkedin { color: #0A66C2; }
        .fa-github { color: inherit; }
        .fa-globe { color: #00BFFF; }
    </style>
    <div class="social-container">
        <a class="social-link" href="https://www.linkedin.com/in/mitesh-kumar0707/" target="_blank">
            <i class="fab fa-linkedin fa-lg"></i> LinkedIn
        </a>
        <a class="social-link" href="https://github.com/miteshkumar07" target="_blank">
            <i class="fab fa-github fa-lg"></i> GitHub
        </a>
        <a class="social-link" href="https://miteshkumar.com/" target="_blank">
            <i class="fas fa-globe fa-lg"></i> Portfolio
        </a>
    </div>
    """, unsafe_allow_html=True)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "⚖️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("docs"):
            with st.expander("📚 View Legal Citations"):
                for doc in message["docs"]:
                    st.markdown(f"**(Source: {doc.metadata.get('source', 'N/A')}, Section: {doc.metadata.get('section', 'N/A')})**")
                    st.info(doc.page_content)

# React to user input
if user_query := st.chat_input("E.g., What are the specific penalties under Article 5?"):
    # Display user message in chat message container
    st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.status("Analyzing request...", expanded=True) as status:
        status.update(label="Initializing models from local cache (First query only)...", state="running")
        agent_app = load_agent()
        
        # Prepare chat history string
        history_str = ""
        for msg in st.session_state.messages[:-1]:
            role = "Human" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        # Node name mapping for a cleaner UI
        node_names = {
            "retriever": "🔍 Searching Legal Database",
            "generated_answer": "✍️ Drafting Legal Response",
            "check_hallucination": "🛡️ Verifying Facts",
            "rewrite_question": "🔄 Re-optimizing Query",
            "clarification_node": "❓ Asking for Clarification",
            "chat_node": "💬 Processing Chat"
        }

        # 1. Run the Agentic LangGraph workflow and stream the thought process
        final_state = {}
        last_printed_action = None
        for output in agent_app.stream({
            "question": user_query, 
            "chat_history": history_str
        }):
            for key, value in output.items():
                friendly_name = node_names.get(key, key)
                # Stream the intermediate step to the UI!
                status.update(label=f"Agent working... ( {friendly_name})", state="running")
                
                # Only print the action if it's different from the last one to prevent spam
                if friendly_name != last_printed_action:
                    st.write(f"⚙️ Action: **{friendly_name}**")
                    last_printed_action = friendly_name
                
                # Accumulate the state updates
                for k, v in value.items():
                    final_state[k] = v
        
        # Collapse the thought process block when finished
        status.update(label="Response generated!", state="complete", expanded=False)
        
    response = final_state.get("generated_answer", "Sorry, I could not generate an answer.")
    all_docs = final_state.get("documents", [])
    
    # 2. Extract Document IDs cited by the LLM (handles [Doc 1] or [Doc 1, Doc 2])
    import re
    cited_doc_indices = set()
    for match in re.finditer(r'\[Docs?\s*([0-9,\sDoc]+)\]', response, re.IGNORECASE):
        nums = re.findall(r'\d+', match.group(1))
        for num_str in nums:
            doc_num = int(num_str)
            # 1-based index in the prompt, so subtract 1 for array index
            if 1 <= doc_num <= len(all_docs):
                cited_doc_indices.add(doc_num - 1)
            
    # If the LLM cited specific docs, only show those!
    # If it cited nothing (e.g. refused to answer), we show no docs.
    # But if the response didn't have [Doc N] for some reason, fallback to showing all of them.
    if cited_doc_indices:
        cited_docs = [all_docs[i] for i in sorted(list(cited_doc_indices))]
    elif not all_docs:
        cited_docs = []
    elif any(phrase in response.lower() for phrase in ["not provided", "do not know", "cannot", "apologize", "sorry"]):
        cited_docs = [] # Refused to answer due to missing info
    else:
        cited_docs = all_docs # Fallback
    
    # Format the citations in the response to be cleaner superscripts
    clean_response = response
    
    # First, entirely strip out the [Source: ... | Section: ...] blocks
    clean_response = re.sub(r'\[Source:.*?\| Section:.*?\]', '', clean_response, flags=re.IGNORECASE)
    
    # Next, replace [Doc 2, Doc 7] or [Doc 9] with <sup>[2, 7]</sup>
    def replace_doc_refs(match):
        nums = re.findall(r'\d+', match.group(1))
        return f"<sup>[{', '.join(nums)}]</sup>"
        
    clean_response = re.sub(r'\[Docs?\s*([0-9,\sDoc]+)\]', replace_doc_refs, clean_response, flags=re.IGNORECASE)
    
    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown(clean_response, unsafe_allow_html=True)
        
        # 3. Display ONLY the Cited Documents in the expander (Cleaner formatting)
        if cited_docs and len(cited_docs) > 0:
            with st.expander("📚 View Legal Citations"):
                for idx, doc in enumerate(cited_docs):
                    # Find the actual original document number that the LLM cited
                    # We have to reverse-lookup from cited_doc_indices
                    original_doc_num = sorted(list(cited_doc_indices))[idx] + 1
                    st.markdown(f"**[{original_doc_num}]**")
                    
                    # Clean up weird markdown artifacts like (<sup>58</sup> ) from the legal text
                    clean_text = re.sub(r'\(<sup[^>]*>.*?</sup>\s*\)', '', doc.page_content)
                    
                    # Convert massive headers (#) to smaller headers (####) so they don't overpower the UI
                    clean_text = re.sub(r'^#+\s*', '#### ', clean_text, flags=re.MULTILINE)
                    
                    # Use st.info to ensure Markdown renders correctly, but styled via CSS
                    st.info(clean_text)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": clean_response, "docs": cited_docs})
