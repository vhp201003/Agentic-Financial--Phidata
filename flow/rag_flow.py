# rag_flow.py
from utils.logging import setup_logging
from phi.agent import RunResponse

logger = setup_logging()

def rag_flow(sub_query: str, rag_agent, rag_tool) -> str:
    """Xử lý flow của RAG: gọi rag_tool để lấy tài liệu, sau đó gọi rag_agent để tóm tắt."""
    try:
        logger.info(f"Executing RAG query: {sub_query}")

        # Gọi rag_tool để lấy nội dung tài liệu
        document_content = rag_tool.run(sub_query)
        logger.debug(f"Document content from rag_tool: {document_content[:100]}...")

        # Nếu không tìm thấy tài liệu, trả về thông báo lỗi từ rag_tool
        if document_content.startswith("Không tìm thấy tài liệu") or document_content.startswith("Lỗi khi truy xuất tài liệu"):
            logger.warning(f"No relevant documents found for query: {sub_query}")
            return document_content

        # Gửi nội dung tài liệu cho rag_agent để tóm tắt
        rag_response = rag_agent.run(sub_query + "\n\nDocument content:\n" + document_content)
        if isinstance(rag_response, RunResponse):
            rag_response = rag_response.content
        
        logger.debug(f"Summary from rag_agent: {rag_response}")
        return rag_response

    except Exception as e:
        logger.error(f"Error in rag_flow: {str(e)}")
        return f"Không tìm thấy tài liệu liên quan đến {sub_query}. Lỗi: {str(e)}"