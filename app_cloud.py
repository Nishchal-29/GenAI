import streamlit as st
import time
from datetime import datetime
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

st.set_page_config(
    page_title="DeepSeek Code Companion",
    page_icon="🧠",
    layout="wide"
)

THEMES = {
    "Dark Mode": {
        "bg_color": "#1a1a1a",
        "sidebar_bg": "#2d2d2d",
        "text_color": "#ffffff",
        "accent_color": "#4287f5",
        "secondary_bg": "#3d3d3d",
        "card_bg": "#2a2a2a",
        "code_bg": "#252525",
        "highlight_color": "#f9a825"
    },
    "Oceanic": {
        "bg_color": "#0f2027",
        "sidebar_bg": "#203a43",
        "text_color": "#f0f8ff",
        "accent_color": "#64b5f6",
        "secondary_bg": "#2c5364",
        "card_bg": "#1c313a",
        "code_bg": "#162025",
        "highlight_color": "#4fc3f7"
    },
    "Forest": {
        "bg_color": "#0a1e0f",
        "sidebar_bg": "#1e3b26",
        "text_color": "#e8f5e9",
        "accent_color": "#66bb6a",
        "secondary_bg": "#2e5735",
        "card_bg": "#1b2e20",
        "code_bg": "#0f2213",
        "highlight_color": "#aed581"
    },
    "Sunset": {
        "bg_color": "#1a0f1c",
        "sidebar_bg": "#2c1e33",
        "text_color": "#ffebee",
        "accent_color": "#ff7043",
        "secondary_bg": "#3e2842",
        "card_bg": "#261829",
        "code_bg": "#1c131e",
        "highlight_color": "#ffab40"
    }
}

LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", 
    "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin"
]

HF_MODELS = {
    "Bloom": "bigscience/bloom", 
    "Bloom-560m": "bigscience/bloom-560m",
    "Flan-T5-XL": "google/flan-t5-xl",
    "Flan-T5-Large": "google/flan-t5-large",
    "GPT2": "gpt2",
    "GPT2-XL": "gpt2-xl",
    "GPT-Neo": "EleutherAI/gpt-neo-1.3B"
}

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

if "theme" not in st.session_state:
    st.session_state.theme = "Dark Mode"

if "language" not in st.session_state:
    st.session_state.language = "Python"

if "thinking_visible" not in st.session_state:
    st.session_state.thinking_visible = True
    
if "model" not in st.session_state:
    st.session_state.model = "GPT2" 

theme = THEMES[st.session_state.theme]

st.markdown(f"""
<style>
    .main {{
        background-color: {theme["bg_color"]};
        color: {theme["text_color"]};
    }}
    
    .sidebar .sidebar-content {{
        background-color: {theme["sidebar_bg"]};
    }}
    
    .stTextInput textarea, .stTextArea textarea {{
        color: {theme["text_color"]} !important;
        background-color: {theme["secondary_bg"]} !important;
        border: 1px solid {theme["accent_color"]}40 !important;
    }}
    
    .stSelectbox div[data-baseweb="select"] {{
        color: {theme["text_color"]} !important;
        background-color: {theme["secondary_bg"]} !important;
        border: 1px solid {theme["accent_color"]}40 !important;
    }}
    
    .stSelectbox svg {{
        fill: {theme["text_color"]} !important;
    }}
    
    div[role="listbox"] div {{
        background-color: {theme["sidebar_bg"]} !important;
        color: {theme["text_color"]} !important;
    }}
    
    .stButton > button {{
        background-color: {theme["accent_color"]};
        color: {theme["text_color"]};
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }}
    
    .stButton > button:hover {{
        background-color: {theme["highlight_color"]};
    }}
    
    .message-container {{
        background-color: {theme["card_bg"]};
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid {theme["accent_color"]};
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    
    .thinking-container {{
        background-color: {theme["card_bg"]}90;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 10px 0;
        border-left: 3px dashed {theme["highlight_color"]};
    }}
    
    .time-stamp {{
        font-size: 0.7rem;
        color: {theme["text_color"]}80;
        margin-top: 5px;
        font-style: italic;
    }}
    
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }}
    
    .code-block {{
        background-color: {theme["code_bg"]};
        border-radius: 5px;
        border-left: 3px solid {theme["accent_color"]};
    }}
    
    /* Hide default streamlit branding */
    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {theme["bg_color"]};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {theme["accent_color"]}80;
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {theme["accent_color"]};
    }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
        <div style='text-align: center;'>
            <h2 style='color: {theme["accent_color"]}; margin-bottom: 5px;'>🧠 DeepSeek</h2>
            <p style='font-size: 1.2em; font-weight: 500; margin-top: 0;'>Code Companion</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    
    selected_theme = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme)
    )
    
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    selected_model = st.selectbox(
        "Model",
        list(HF_MODELS.keys()),
        index=list(HF_MODELS.keys()).index(st.session_state.model) if st.session_state.model in HF_MODELS else 0
    )
    
    if selected_model != st.session_state.model:
        st.session_state.model = selected_model
    
    selected_language = st.selectbox(
        "Code Output Language",
        LANGUAGES,
        index=LANGUAGES.index(st.session_state.language)
    )
    
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
    
    st.session_state.thinking_visible = st.toggle("Show Thinking Process", value=st.session_state.thinking_visible)
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    st.divider()
    
    st.markdown("### 💡 Model Capabilities")
    capabilities = [
        "🐍 Code Generation & Completion",
        "🐞 Error Detection & Debugging",
        "📝 Code Documentation",
        "🔧 Code Refactoring",
        "📊 Algorithm Explanation",
        "💻 Multi-language Support"
    ]
    
    for capability in capabilities:
        st.markdown(f"- {capability}")
    
    st.divider()
    
    st.markdown("### 🚀 Try These Examples")
    if st.button("Debug a recursive function"):
        example_query = "Debug this recursive function that's causing a stack overflow: def factorial(n): return n * factorial(n-1)"
        st.session_state.message_log.append({"role": "user", "content": example_query})
        st.rerun()
        
    if st.button("Explain a sorting algorithm"):
        example_query = "Explain how quicksort works and show me an implementation in Python"
        st.session_state.message_log.append({"role": "user", "content": example_query})
        st.rerun()
        
    if st.button("Generate a REST API"):
        example_query = f"Create a simple REST API for a todo app using {st.session_state.language}"
        st.session_state.message_log.append({"role": "user", "content": example_query})
        st.rerun()
    
    st.divider()
    
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Select your theme** and preferred language above
        2. **Choose a model** from the dropdown menu
        3. **Type your coding question** in the chat input
        4. **View the AI's response** with syntax-highlighted code
        5. **Adjust the temperature** for different response styles
        """)
    
    st.divider()
    
    st.markdown("""
        <div style='font-size: 0.8em; opacity: 0.8; text-align: center;'>
            Built with <a href='https://huggingface.co/' style='color: inherit; text-decoration: underline;'>Hugging Face</a> | 
            <a href='https://streamlit.io/' style='color: inherit; text-decoration: underline;'>Streamlit</a>
        </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_hf_api_client():
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        st.warning("⚠️ HF_TOKEN not found in environment. Some models may have limited access.")
    
    return InferenceClient(token=hf_token)

def generate_response_with_hf_api(client, query, model_name, temp):
    try:
        model_id = HF_MODELS[model_name]
        system_prompt = f"You are DeepSeek, an expert AI coding assistant specialized in {st.session_state.language}."
        
        conversation = system_prompt + "\n\n"
        
        for msg in st.session_state.message_log[-5:]: 
            if msg["role"] == "user":
                conversation += f"User: {msg['content']}\n\n"
            elif msg["role"] == "ai":
                conversation += f"Assistant: {msg['content']}\n\n"
        
        conversation += f"User: {query}\n\nAssistant:"
        
        try:
            if "t5" in model_id.lower():
                response = client.text_generation(
                    prompt=conversation,
                    model=model_id,
                    max_new_tokens=512,
                    temperature=temp,
                    do_sample=True
                )
            else:
                response = client.text_generation(
                    prompt=conversation,
                    model=model_id,
                    max_new_tokens=512,
                    temperature=temp,
                    do_sample=True
                )
            
            return response.strip()
        except Exception as e:
            fallback_model = "gpt2"
            st.warning(f"Error with primary model: {str(e)}. Trying fallback model...")
            
            response = client.text_generation(
                prompt=conversation,
                model=fallback_model,
                max_new_tokens=512, 
                temperature=temp,
                do_sample=True
            )
            
            return response.strip() + "\n\n(Note: This response was generated using a fallback model)"
    except Exception as e:
        return f"""
I'm sorry, I encountered an error generating a response: {str(e)}

This could be due to:
1. The model being unavailable on Hugging Face
2. Missing or invalid HF_TOKEN
3. Network connectivity issues
4. Rate limiting on the Hugging Face API

Please try:
- Selecting a different model
- Checking your HF_TOKEN in the .env file
- Reducing the complexity of your query
- Waiting a few minutes and trying again
"""

def format_code_block(content):
    formatted = content
    if "```" in content:
        parts = content.split("```")
        for i in range(1, len(parts), 2):
            if i < len(parts):
                code_lines = parts[i].strip().split("\n")
                if code_lines[0] and not code_lines[0].startswith(" "):
                    lang = code_lines[0]
                    code = "\n".join(code_lines[1:])
                    parts[i] = f"{lang}\n{code}"
                
                parts[i] = f'<div class="code-block">\n```{parts[i]}```\n</div>'
        
        formatted = "".join(parts)
    
    return formatted

def generate_thinking_process(query):
    thinking_steps = [
        "Analyzing the problem...",
        f"Considering best practices for {st.session_state.language}...",
        "Evaluating potential solutions...",
        "Checking for edge cases...",
        "Optimizing for readability and performance...",
        "Formulating explanation and examples..."
    ]
    
    thinking_container = st.empty()
    
    for step in thinking_steps:
        timestamp = datetime.now().strftime("%H:%M:%S")
        thinking_container.markdown(f"""
        <div class="thinking-container">
            <span>💭 {step}</span>
            <div class="time-stamp">⏱️ {timestamp}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.5 + (hash(step) % 7) / 10)
    
    return thinking_container

st.markdown("""
    <div class='header-container'>
        <h1>💻 Code With Confidence</h1>
    </div>
""", unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with st.chat_message(message["role"]):
            if message["role"] == "ai":
                formatted_message = format_code_block(message["content"])
                st.markdown(formatted_message, unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
            
            st.markdown(f"""
            <div class="time-stamp">⏱️ {timestamp}</div>
            """, unsafe_allow_html=True)

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    st.warning("⚠️ HF_TOKEN not found in your .env file. Some models may have usage limits without authentication.")
    with st.expander("How to set up your Hugging Face API token"):
        st.markdown("""
        ### Setting up your Hugging Face API token
        
        1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
        2. Go to your profile settings and create an API token
        3. Create a `.env` file in your project directory
        4. Add the following line to your `.env` file:
           ```
           HF_TOKEN=your-token-here
           ```
        5. Restart the application
        
        This will authenticate your requests and increase your rate limits.
        """)

user_query = st.chat_input("Ask me about code, debugging, documentation, or algorithms...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div class="time-stamp">⏱️ {timestamp}</div>
        """, unsafe_allow_html=True)
    
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    if st.session_state.thinking_visible:
        thinking_placeholder = generate_thinking_process(user_query)
    
    with st.spinner("🧠 Processing..."):
        client = get_hf_api_client()
        ai_response = generate_response_with_hf_api(
            client, 
            user_query, 
            st.session_state.model, 
            temperature
        )
    
    if st.session_state.thinking_visible:
        thinking_placeholder.empty()
    
    with st.chat_message("ai"):
        formatted_response = format_code_block(ai_response)
        st.markdown(formatted_response, unsafe_allow_html=True)
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div class="time-stamp">⏱️ {timestamp}</div>
        """, unsafe_allow_html=True)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})