# rag_flow.py
from utils.logging import setup_logging

logger = setup_logging()

def rag_flow(sub_query: str, rag_tool, tickers: list = None, company: str = None) -> list:
    """Xử lý flow của RAG: gọi rag_tool để lấy tài liệu, trả về danh sách tài liệu gốc."""
    try:
        logger.info(f"Executing RAG query: {sub_query}, tickers: {tickers}, company: {company}")
        # Gọi rag_tool với sub_query, tickers và company
        documents = rag_tool.run(sub_query, company=company, tickers=tickers)
        logger.debug(f"Documents from rag_tool: {str(documents)[:100]}...")

        # Kiểm tra lỗi từ rag_tool
        if isinstance(documents, list) and documents and "error" in documents[0]:
            logger.warning(f"No relevant documents found for query: {sub_query}")
            return documents

        logger.info(f"Retrieved {len(documents)} documents")
        return documents

    except Exception as e:
        logger.error(f"Error in rag_flow: {str(e)}")
        return [{"error": f"No relevant financial report information found for {sub_query}. Error: {str(e)}"}]