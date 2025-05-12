import os
import sys
from pathlib import Path
from typing import List, Dict
import gradio as gr
import requests
from utils.logging import setup_logging

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

logger = setup_logging()

# Địa chỉ FastAPI
API_URL = "http://localhost:8000"

# Ánh xạ từ giá trị dropdown sang endpoint FastAPI
AGENT_ENDPOINTS = {
    "RAG Agent": "rag",
    "TextToSQL Agent": "sql",
    "Finance Agent": "finance",
    "Agent Team": "team"
}

# CSS tùy chỉnh để tạo giao diện giống Grok và full screen
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

def call_agent(query: str, agent_type: str, chat_history: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], str]:
    """Gọi agent và cập nhật lịch sử chat."""
    if not query.strip():
        chat_history.append({"role": "assistant", "content": "Vui lòng nhập truy vấn."})
        return chat_history, ""
    
    try:
        endpoint_name = AGENT_ENDPOINTS.get(agent_type)
        if not endpoint_name:
            error_msg = f"Lỗi: Loại agent không hợp lệ {agent_type}"
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": error_msg})
            return chat_history, ""
        
        endpoint = f"{API_URL}/{endpoint_name}"
        response = requests.post(endpoint, json={"query": query})
        response.raise_for_status()
        result = response.json()
        
        # Xử lý phản hồi để chỉ lấy nội dung chuỗi
        response_text = result.get("response", result.get("error", "Không có phản hồi"))
        if isinstance(response_text, dict):
            # Nếu phản hồi là dictionary, lấy trường content
            response_text = response_text.get("content", str(response_text))
        
        # Thêm vào lịch sử chat
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response_text})
        return chat_history, ""
    
    except Exception as e:
        logger.error(f"Error calling {agent_type} Agent: {str(e)}")
        error_msg = f"Lỗi: {str(e)}"
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history, ""

def clear_chat() -> List[Dict[str, str]]:
    """Xóa lịch sử chat."""
    return []

def create_gradio_interface():
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Monochrome(), title="Financial AI Agent") as demo:
        gr.Markdown("# Financial AI Agent")
        gr.Markdown("### Trò chuyện với các agent tài chính của bạn")
        
        # Chatbot hiển thị lịch sử hội thoại
        chatbot = gr.Chatbot(
            label="Lịch sử trò chuyện",
            elem_classes=["chatbot"],
            type="messages",  # Sử dụng định dạng openai-style
            render_markdown=True,
            height="100%"  # Để chatbot chiếm toàn bộ không gian dọc còn lại
        )
        
        with gr.Row(elem_classes=["input-container"]):
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    placeholder="Nhập truy vấn của bạn...",
                    show_label=False,
                    container=False
                )
            with gr.Column(scale=1):
                agent_dropdown = gr.Dropdown(
                    choices=["Agent Team", "TextToSQL Agent", "Finance Agent", "RAG Agent"],
                    value="Agent Team",
                    show_label=False,
                    elem_id="agent-dropdown"
                )
        
        with gr.Row():
            submit_button = gr.Button("Gửi", elem_id="submit-btn")
            clear_button = gr.Button("Xóa lịch sử", elem_id="clear-btn")
        
        # Liên kết sự kiện
        submit_button.click(
            fn=call_agent,
            inputs=[query_input, agent_dropdown, chatbot],
            outputs=[chatbot, query_input]
        )
        clear_button.click(
            fn=clear_chat,
            inputs=None,
            outputs=chatbot
        )
        query_input.submit(
            fn=call_agent,
            inputs=[query_input, agent_dropdown, chatbot],
            outputs=[chatbot, query_input]
        )
    
    return demo

if __name__ == "__main__":
    logger.info("Starting Gradio UI")
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861)