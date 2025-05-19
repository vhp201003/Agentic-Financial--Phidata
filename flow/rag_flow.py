# rag_flow.py
from utils.logging import setup_logging
from phi.agent import RunResponse

logger = setup_logging()

def rag_flow(sub_query: str, rag_agent, rag_tool) -> dict:
    """Xử lý flow của RAG: gọi rag_tool để lấy tài liệu, sau đó gọi rag_agent để tóm tắt."""
    try:
        logger.info(f"Executing RAG query: {sub_query}")

        # Gọi rag_tool để lấy nội dung tài liệu và metadata
        documents = rag_tool.run(sub_query)
        logger.debug(f"Documents from rag_tool: {documents[:100]}...")

        # Kiểm tra lỗi từ rag_tool
        if isinstance(documents, list) and documents and "error" in documents[0]:
            logger.warning(f"No relevant documents found for query: {sub_query}")
            return {"summary": documents[0]["error"], "documents": documents}

        # Chuẩn bị nội dung tài liệu để gửi cho rag_agent
        document_content = "\n\n".join([doc["document"] for doc in documents])
        logger.debug(f"Document content for rag_agent: {document_content[:100]}...")

        # Gửi nội dung tài liệu cho rag_agent để tóm tắt
        rag_response = rag_agent.run(sub_query + "\n\nDocument content:\n" + document_content)
        if isinstance(rag_response, RunResponse):
            rag_response = rag_response.content
        
        logger.debug(f"Summary from rag_agent: {rag_response}")
        return {"summary": rag_response, "documents": documents}

    except Exception as e:
        logger.error(f"Error in rag_flow: {str(e)}")
        return {"summary": f"No relevant financial report information found for {sub_query}. Error: {str(e)}", "documents": []}