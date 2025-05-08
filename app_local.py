import streamlit as st
import time
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

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

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

if "theme" not in st.session_state:
    st.session_state.theme = "Dark Mode"

if "language" not in st.session_state:
    st.session_state.language = "Python"

if "thinking_visible" not in st.session_state:
    st.session_state.thinking_visible = True

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
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    
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
        "📊 Diagram & Algorithm Explanation",
        "💻 Multi-language Support"
    ]
    
    for capability in capabilities:
        st.markdown(f"- {capability}")
    
    st.divider()
    
    st.markdown("""
        <div style='font-size: 0.8em; opacity: 0.8; text-align: center;'>
            Built with <a href='https://ollama.ai/' style='color: inherit; text-decoration: underline;'>Ollama</a> | 
            <a href='https://python.langchain.com/' style='color: inherit; text-decoration: underline;'>LangChain</a> | 
            <a href='https://streamlit.io/' style='color: inherit; text-decoration: underline;'>Streamlit</a>
        </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_llm_engine(model_name, temp):
    return ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=temp
    )

llm_engine = get_llm_engine(selected_model, temperature)

def get_system_prompt():
    return SystemMessagePromptTemplate.from_template(
        f"""You are DeepSeek, an expert AI coding assistant specialized in {st.session_state.language}.
        Provide concise, correct solutions with strategic print statements for debugging.
        When generating code solutions, prioritize writing in {st.session_state.language} unless specifically asked otherwise.
        Break down complex problems step-by-step with clear explanations.
        Always respond in English and follow best practices for {st.session_state.language} development."""
    )

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

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain(current_query=None):
    prompt_sequence = [get_system_prompt()]
    
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    
    if current_query:
        prompt_sequence.append(HumanMessagePromptTemplate.from_template(current_query))
        
    return ChatPromptTemplate.from_messages(prompt_sequence)

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
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
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