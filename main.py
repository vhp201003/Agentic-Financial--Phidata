import sys
from pathlib import Path
from utils.logging import setup_logging
from agents.orchestrator import create_orchestrator
from agents.text_to_sql_agent import create_text_to_sql_agent
from agents.rag_agent import create_rag_agent
from agents.chat_completion_agent import create_chat_completion_agent  # Đảm bảo import đúng
from tools.sql_tool import CustomSQLTool
from tools.rag_tool import CustomRAGTool
from flow.orchestrator_flow import orchestrator_flow

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

logger = setup_logging()

def main():
    """Chạy vòng lặp tương tác để test hệ thống orchestrator với dashboard."""
    logger.info("Starting orchestrator test")
    try:
        orchestrator = create_orchestrator()
        sql_agent = create_text_to_sql_agent()
        rag_agent = create_rag_agent()
        chat_completion_agent = create_chat_completion_agent()  # Sửa: Gọi hàm để tạo instance
        sql_tool = CustomSQLTool()
        rag_tool = CustomRAGTool()
    except Exception as e:
        logger.error(f"Failed to create orchestrator or agents: {str(e)}")
        print(f"Error: Không thể khởi tạo orchestrator hoặc agents: {str(e)}")
        return

    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            logger.info("Exiting orchestrator test")
            break
        if not query:
            print("Please enter a valid query.")
            continue

        # Gọi orchestrator flow
        result = orchestrator_flow(query, orchestrator, sql_agent, sql_tool, rag_agent, rag_tool, chat_completion_agent)
        
        # In kết quả
        print("\nCâu trả lời:")
        print(result.get("message", "Không có câu trả lời."))

if __name__ == "__main__":
    main()