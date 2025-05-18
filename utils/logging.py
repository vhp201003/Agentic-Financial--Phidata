import logging
from io import StringIO
from queue import Queue

# Biến toàn cục để lưu log
log_stream = StringIO()
log_queue = Queue()

def setup_logging():
    logger = logging.getLogger("AgentTeam")
    logger.setLevel(logging.INFO)
    
    # Handler cho console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Handler cho StringIO (để lấy log cuối cùng nếu cần)
    string_handler = logging.StreamHandler(log_stream)
    string_handler.setLevel(logging.INFO)
    string_handler.setFormatter(formatter)
    
    # Handler để gửi log vào log_queue
    class QueueHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            log_queue.put(log_entry)
    
    queue_handler = QueueHandler()
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(formatter)
    
    # Thêm các handler
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(string_handler)
        logger.addHandler(queue_handler)
    
    return logger

def get_collected_logs():
    """Lấy log đã thu thập và reset stream."""
    logs = log_stream.getvalue()
    log_stream.truncate(0)
    log_stream.seek(0)
    return logs