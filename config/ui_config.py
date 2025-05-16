import sys
from pathlib import Path
BASE_DIR = Path(__file__)
sys.path.append(str(BASE_DIR))


# Configuration settings for the Financial AI Agent

# API configuration
API_URL = "http://localhost:8010"

# Agent endpoint mapping
AGENT_ENDPOINTS = {
    "RAG Agent": "rag",
    "TextToSQL Agent": "sql",
    "Finance Agent": "finance",
    "Agent Team": "team"
}

# Custom CSS for styling the Gradio interface
CUSTOM_CSS = """
body {
    background-color: #1a1a1a;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    margin: 0;
    padding: 0;
    height: 100vh;
    overflow: hidden;
}
.gradio-container {
    width: 100%;
    height: 100vh;
    margin: 0;
    padding: 20px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
}
.chatbot {
    flex-grow: 1;
    width: 100%;
    overflow-y: auto;
    margin-bottom: 20px;
}
.chatbot .message {
    border-radius: 15px;
    padding: 12px 16px;
    margin: 8px 12px;
    max-width: 80%;
}
.chatbot .user {
    background-color: #007bff;
    color: white;
    margin-left: auto;
}
.chatbot .bot {
    background-color: #2d2d2d;
    color: #e0e0e0;
    margin-right: auto;
}
.chatbot .error {
    background-color: #ff4d4d;
    color: white;
}
.input-container {
    background-color: #2d2d2d;
    border-radius: 10px;
    padding: 10px;
    width: 100%;
    box-sizing: border-box;
}
#agent-dropdown {
    background-color: #333;
    color: #e0e0e0;
    border: none;
}
#submit-btn, #clear-btn {
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 8px;
}
#submit-btn:hover, #clear-btn:hover {
    background-color: #0056b3;
}
h1, h3 {
    color: #e0e0e0;
    text-align: center;
    margin: 10px 0;
}
"""
