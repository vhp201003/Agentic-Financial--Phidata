# import os
# import sys
# from pathlib import Path
# import gradio as gr
# import requests
# from utils.logging import setup_logging
# # from config import API_URL, AGENT_ENDPOINTS, CUSTOM_CSS
# from config.ui_config import API_URL, AGENT_ENDPOINTS, CUSTOM_CSS
# # Add project root to sys.path
# BASE_DIR = Path(__file__).resolve().parent
# sys.path.append(str(BASE_DIR))

# logger = setup_logging()

# class FinancialAIAgent:
#     """Main class for handling the Financial AI Agent functionality."""
    
#     def __init__(self, api_url=API_URL, agent_endpoints=AGENT_ENDPOINTS):
#         """Initialize the Financial AI Agent with API configuration."""
#         self.api_url = api_url
#         self.agent_endpoints = agent_endpoints
#         self.logger = logger
    
#     def call_agent(self, query, agent_type, chat_history):
#         """Call the appropriate agent API and update chat history."""
#         if not query.strip():
#             chat_history.append({"role": "assistant", "content": "Vui lòng nhập truy vấn."})
#             return chat_history, ""
        
#         try:
#             endpoint_name = self.agent_endpoints.get(agent_type)
#             if not endpoint_name:
#                 error_msg = f"Lỗi: Loại agent không hợp lệ {agent_type}"
#                 chat_history.append({"role": "user", "content": query})
#                 chat_history.append({"role": "assistant", "content": error_msg})
#                 return chat_history, ""
            
#             endpoint = f"{self.api_url}/{endpoint_name}"
#             response = requests.post(endpoint, json={"query": query})
#             response.raise_for_status()
#             result = response.json()
            
#             # Process response to extract string content
#             response_text = result.get("response", result.get("error", "Không có phản hồi"))
#             if isinstance(response_text, dict):
#                 response_text = response_text.get("content", str(response_text))
            
#             # Update chat history
#             chat_history.append({"role": "user", "content": query})
#             chat_history.append({"role": "assistant", "content": response_text})
#             return chat_history, ""
        
#         except Exception as e:
#             self.logger.error(f"Error calling {agent_type} Agent: {str(e)}")
#             error_msg = f"Lỗi: {str(e)}"
#             chat_history.append({"role": "user", "content": query})
#             chat_history.append({"role": "assistant", "content": error_msg})
#             return chat_history, ""
    
#     @staticmethod
#     def clear_chat():
#         """Clear the chat history."""
#         return []
    
#     def create_ui(self):
#         """Create and configure the Gradio interface."""
#         with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Monochrome(), title="Financial AI Agent") as demo:
#             gr.Markdown("# Financial AI Agent")
#             gr.Markdown("### Trò chuyện với các agent tài chính của bạn")
            
#             # Chat display
#             chatbot = gr.Chatbot(
#                 label="Lịch sử trò chuyện",
#                 elem_classes=["chatbot"],
#                 type="messages",
#                 render_markdown=True,
#                 height="100%"
#             )
            
#             # Input area
#             with gr.Row(elem_classes=["input-container"]):
#                 with gr.Column(scale=4):
#                     query_input = gr.Textbox(
#                         placeholder="Nhập truy vấn của bạn...",
#                         show_label=False,
#                         container=False
#                     )
#                 with gr.Column(scale=1):
#                     agent_dropdown = gr.Dropdown(
#                         choices=list(self.agent_endpoints.keys()),
#                         value="Agent Team",
#                         show_label=False,
#                         elem_id="agent-dropdown"
#                     )
            
#             # Action buttons
#             with gr.Row():
#                 submit_button = gr.Button("Gửi", elem_id="submit-btn")
#                 clear_button = gr.Button("Xóa lịch sử", elem_id="clear-btn")
            
#             # Event handlers
#             submit_button.click(
#                 fn=self.call_agent,
#                 inputs=[query_input, agent_dropdown, chatbot],
#                 outputs=[chatbot, query_input]
#             )
            
#             clear_button.click(
#                 fn=self.clear_chat,
#                 inputs=None,
#                 outputs=chatbot
#             )
            
#             query_input.submit(
#                 fn=self.call_agent,
#                 inputs=[query_input, agent_dropdown, chatbot],
#                 outputs=[chatbot, query_input]
#             )
        
#         return demo
    
#     def launch(self, server_name="0.0.0.0", server_port=7861):
#         """Launch the Gradio interface."""
#         self.logger.info("Starting Gradio UI")
#         demo = self.create_ui()
#         demo.launch(server_name=server_name, server_port=server_port)

# def main():
#     """Main entry point for the application."""
#     agent = FinancialAIAgent()
#     agent.launch()

# if __name__ == "__main__":
#     main()
# ui.py
import gradio as gr
from Data_Platform.Final.financial_agent_system.agents.orchestrator import create_agent_team
from utils.logging import setup_logging
import json

logger = setup_logging()

agent_team = create_agent_team()

def process_query(query):
    try:
        result_json = agent_team.run(query)
        result_dict = json.loads(result_json)
        if result_dict["status"] == "success":
            return result_dict["data"]["result"]
        else:
            return f"Error: {result_dict['message']}"
    except Exception as e:
        logger.error(f"Error in UI: {str(e)}")
        return f"Error: {str(e)}"

with gr.Blocks(title="Financial Assistant") as demo:
    gr.Markdown("# Financial Assistant")
    query_input = gr.Textbox(label="Enter your query", placeholder="e.g., What is the stock price of Apple?")
    output = gr.Textbox(label="Response")
    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=process_query, inputs=query_input, outputs=output)

demo.launch(server_port=7861)