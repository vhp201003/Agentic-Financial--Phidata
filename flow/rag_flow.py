# rag_flow.py
from utils.logging import setup_logging

logger = setup_logging()

def rag_flow(sub_query: str, rag_tool) -> list:
    """Xử lý flow của RAG: gọi rag_tool để lấy tài liệu, trả về danh sách tài liệu gốc."""
    try:
        logger.info(f"Executing RAG query: {sub_query}")

        # Gọi rag_tool để lấy nội dung tài liệu và metadata
        documents = rag_tool.run(sub_query)
        logger.debug(f"Documents from rag_tool: {documents[:100]}...")

        # Kiểm tra lỗi từ rag_tool
        if isinstance(documents, list) and documents and "error" in documents[0]:
            logger.warning(f"No relevant documents found for query: {sub_query}")
            return documents

        logger.debug(f"Retrieved documents: {documents}")
        return documents

    except Exception as e:
        logger.error(f"Error in rag_flow: {str(e)}")
        return [{"error": f"No relevant financial report information found for {sub_query}. Error: {str(e)}"}]