import streamlit as st
import requests
import json
from utils.logging import setup_logging
from config.ui_config import API_URL
from typing import List, Dict
import pandas as pd

logger = setup_logging()

# CSS tùy chỉnh để thiết kế giao diện chat hiện đại
st.markdown("""
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
""", unsafe_allow_html=True)

def create_dashboard(data: dict, visualization: dict):
    """Tạo dashboard trên Streamlit dựa trên dữ liệu."""
    st.markdown("#### Dashboard Kết Quả", unsafe_allow_html=True)
    visualization_type = visualization.get("type", "none")
    required_columns = visualization.get("required_columns", [])

    if visualization_type == "table":
        values = data
        if isinstance(values, list):
            df = pd.DataFrame(values)
            st.dataframe(df, use_container_width=True)
        elif isinstance(values, dict):
            table_data = [[k, v] for k, v in values.items()]
            df = pd.DataFrame(table_data, columns=["Chỉ số", "Giá trị"])
            st.dataframe(df, use_container_width=True)
    elif visualization_type == "time series":
        values = data
        if values and isinstance(values, list) and all("date" in item and "close_price" in item for item in values):
            df = pd.DataFrame(values)
            df["date"] = pd.to_datetime(df["date"])
            st.line_chart(df.set_index("date")["close_price"], use_container_width=True)
        else:
            st.markdown("Dữ liệu không phù hợp để vẽ biểu đồ time series.")
    # Các trường hợp visualization khác...

def process_query(query: str, agent_type: str, chat_history: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], Dict]:
    """Gọi endpoint FastAPI và cập nhật lịch sử chat."""
    if not query.strip():
        chat_history.append({"role": "assistant", "content": "Vui lòng nhập truy vấn."})
        return chat_history, {}
    
    try:
        endpoint = f"{API_URL}/team"
        response = requests.post(endpoint, json={"query": query})
        response.raise_for_status()
        result = response.json()
        
        # Xử lý phản hồi
        result_json = result.get("response")
        result_dict = json.loads(result_json)
        
        if result_dict["status"] == "success":
            response_text = result_dict.get("message", "Không có phản hồi.")
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response_text})

            # Trả về thông tin dashboard nếu có
            dashboard_info = result_dict.get("data", {}).get("dashboard", {})
            return chat_history, dashboard_info
        
        else:
            response_text = f"Error: {result_dict.get('message', 'Unknown error')}"
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response_text})
        
        return chat_history, {}
    
    except Exception as e:
        logger.error(f"Error in UI: {str(e)}")
        error_msg = f"Error: {str(e)}"
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history, {}

# Khởi tạo session state để lưu lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Giao diện Streamlit
st.title("Financial Assistant")
st.markdown("### Trò chuyện với các agent tài chính của bạn")

# Khu vực hiển thị lịch sử chat
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div class="user-message">
                    <div class="message-bubble">{message['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="assistant-message">
                    <div class="message-bubble">{message['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Lồng dashboard ngay sau tin nhắn của assistant nếu có
            if "dashboard_info" in st.session_state and st.session_state.chat_history[-1] == message:
                dashboard_info = st.session_state.get("dashboard_info", {})
                if dashboard_info.get("enabled", False) and dashboard_info.get("data"):
                    with st.container():
                        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                        create_dashboard(dashboard_info["data"], dashboard_info["visualization"])
                        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Thanh nhập liệu cố định ở dưới cùng
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    query_input = st.text_input("Nhập truy vấn của bạn...", key="query_input")
    agent_type = st.selectbox("Chọn agent", ["Agent Team"], key="agent_type")

    # Khu vực nút điều khiển
    col1, col2 = st.columns([1, 1])
    with col1:
        submit_button = st.button("Gửi", key="submit_button")
    with col2:
        clear_button = st.button("Xóa lịch sử", key="clear_button", help="Xóa toàn bộ lịch sử chat")
    st.markdown('</div>', unsafe_allow_html=True)

# Xử lý sự kiện
if submit_button and query_input:
    chat_history, dashboard_info = process_query(query_input, agent_type, st.session_state.chat_history)
    st.session_state.chat_history = chat_history
    st.session_state.dashboard_info = dashboard_info  # Lưu dashboard_info để hiển thị sau tin nhắn assistant
    st.rerun()  # Rerun để cập nhật giao diện

if clear_button:
    st.session_state.chat_history = []
    if "dashboard_info" in st.session_state:
        del st.session_state.dashboard_info
    st.rerun()  # Rerun để cập nhật giao diện

# Tự động cuộn xuống tin nhắn mới nhất
st.markdown(
    """
    <script>
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
    """,
    unsafe_allow_html=True
)