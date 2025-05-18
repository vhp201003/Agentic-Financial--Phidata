import sys
from pathlib import Path
BASE_DIR = Path(__file__)
sys.path.append(str(BASE_DIR))


# Configuration settings for the Financial AI Agent

# API configuration
API_URL = "http://localhost:8000"

# Agent endpoint mapping
AGENT_ENDPOINTS = {
    "RAG Agent": "rag",
    "TextToSQL Agent": "sql",
    "Finance Agent": "finance",
    "Agent Team": "team"
}

# Custom CSS for styling the Gradio interface
CUSTOM_CSS = """
<style>
/* Container chính */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 15px;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    background-color: #f5f7fa;
    margin-bottom: 20px;
}

/* Bong bóng chat */
.user-message {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 15px;
}
.assistant-message {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 15px;
}
.message-bubble {
    max-width: 70%;
    padding: 12px 18px;
    border-radius: 15px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    word-wrap: break-word;
    font-size: 16px;
    line-height: 1.5;
}
.user-message .message-bubble {
    background-color: #4CAF50;
    color: white;
}
.assistant-message .message-bubble {
    background-color: #ffffff;
    color: #333;
    border: 1px solid #e0e0e0;
}

/* Dashboard */
.dashboard-container {
    margin-left: 20px;
    margin-bottom: 15px;
    padding: 10px;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

/* Thanh nhập liệu cố định ở dưới cùng */
.input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 10px 20px;
    background-color: #ffffff;
    border-top: 1px solid #e0e0e0;
    z-index: 1000;
}
.input-container .stTextInput, .input-container .stSelectbox {
    margin-bottom: 10px;
}
.button-container {
    display: flex;
    gap: 10px;
}
.stButton>button {
    background-color: #007bff;
    color: white;
    border-radius: 5px;
    padding: 8px 15px;
}
.stButton>button:hover {
    background-color: #0056b3;
}
.clear-button>button {
    background-color: #dc3545;
}
.clear-button>button:hover {
    background-color: #b02a37;
}

/* Tiêu đề và bố cục tổng thể */
.stTitle {
    color: #333;
    font-weight: 700;
}
.stMarkdown {
    color: #555;
}
</style>
"""