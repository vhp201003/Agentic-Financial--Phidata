import streamlit as st
import requests
import json
import pandas as pd
import markdown
from datetime import datetime
from config.ui_config import API_URL

# Cấu hình
st.set_page_config(layout="wide")
API_URL = API_URL + "/team"
# CSS tùy chỉnh
st.markdown("""
<style>
.chat-dashboard-container {
    display: flex;
    width: 100vw;
    min-height: calc(100vh - 70px);
}
.chat-column {
    flex: 7;
    display: flex;
    flex-direction: column;
    padding: 10px;
}
.chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
}
.input-container {
    height: 80px;
    padding: 10px 0;
}
.dashboard-column {
    flex: 3;
    padding: 10px;
    border-left: 1px solid #e0e0e0;
}
.user-message {
    background-color: #4CAF50;
    color: white;
    margin-left: auto;
    max-width: 70%;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.assistant-message {
    background-color: white;
    border: 1px solid #e0e0e0;
    color: black;
    max-width: 70%;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    font-size: 24px;
    text-align: center;
    line-height: 40px;
    margin: 0 10px;
}
.user-message .user-avatar {
    background-color: #4CAF50;
    color: white;
}
.assistant-message .assistant-avatar {
    background-color: #007bff;
    color: white;
}
@media (max-width: 768px) {
    .chat-dashboard-container {
        flex-direction: column;
    }
    .chat-column, .dashboard-column {
        flex: 1;
        width: 100%;
    }
    .avatar {
        width: 30px;
        height: 30px;
        font-size: 20px;
        line-height: 30px;
    }
    .timestamp {
        font-size: 10px;
    }
    .input-container {
        height: 100px;
    }
    table, th, td {
        font-size: 12px;
        padding: 5px;
    }
}
</style>
""", unsafe_allow_html=True)

def markdown_table_to_html(markdown_text):
    import re
    # Tìm tất cả các bảng trong markdown
    table_pattern = r'\|.*?\|\n\|[-|:\s]+\|\n(?:\|.*?\|\n)*'
    tables = re.findall(table_pattern, markdown_text, re.MULTILINE)
    
    if not tables:
        return markdown.markdown(markdown_text)
    
    html_content = markdown_text
    for table in tables:
        # Tách các dòng trong bảng
        rows = table.strip().split('\n')
        if len(rows) < 2:
            continue
            
        # Tách header và body
        headers = [h.strip() for h in rows[0].split('|') if h.strip()]
        body_rows = [[cell.strip() for cell in row.split('|') if cell.strip()] for row in rows[2:]]
        
        # Tạo HTML table
        html_table = "<table>\n<thead>\n<tr>"
        for header in headers:
            html_table += f"<th>{header}</th>"
        html_table += "</tr>\n</thead>\n<tbody>"
        
        for row in body_rows:
            html_table += "\n<tr>"
            for cell in row:
                html_table += f"<td>{cell}</td>"
            html_table += "</tr>"
        html_table += "\n</tbody>\n</table>"
        
        # Thay thế bảng markdown bằng HTML table
        html_content = html_content.replace(table, html_table)
    
    return markdown.markdown(html_content)


# Khởi tạo session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'dashboard_info' not in st.session_state:
    st.session_state.dashboard_info = None

# Tiêu đề
st.title("Financial Assistant")
st.markdown("Trò chuyện với các agent tài chính của bạn")

# Bố cục chính
chat_col, dashboard_col = st.columns([7, 3])

# Khu vực chat
with chat_col:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; align-items: center;'>
                <div class='user-message'>
                    {markdown.markdown(chat['message'])}
                    <div class='timestamp'>{chat['timestamp']}</div>
                </div>
                <div class='avatar user-avatar'>👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display: flex; align-items: center;'>
                <div class='avatar assistant-avatar'>🤖</div>
                <div class='assistant-message'>
                    {markdown_table_to_html(chat['message'])}  <!-- Sử dụng hàm để render bảng -->
                    <div class='timestamp'>{chat['timestamp']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)   

    # Tự động cuộn
    st.markdown("""
    <script>
        const chatContainer = document.querySelector('.chat-container');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    </script>
    """, unsafe_allow_html=True)

    # Khung nhập liệu
    with st.container():
        query = st.text_input("Nhập câu hỏi của bạn (ví dụ: Giá đóng cửa của Apple ngày 01/01/2025?)", key="query")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Gửi", key="send", type="primary"):
                if query:
                    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                    st.session_state.chat_history.append({"role": "user", "message": query, "timestamp": timestamp})
                    
                    try:
                        response = requests.post(API_URL, json={"query": query})
                        response_data = response.json()
                        print(response_data)
                        response_json = json.loads(response_data['response'])
                        if response_json['status'] == 'success':
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "message": response_json['message'],
                                "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                            })
                            if 'data' in response_json and 'dashboard' in response_json['data'] and response_json['data']['dashboard']['enabled']:
                                st.session_state.dashboard_info = response_json['data']['dashboard']
                            else:
                                st.session_state.dashboard_info = None
                            # Lưu log vào session state để hiển thị
                            st.session_state.logs = response_json.get('logs', 'Không có log nào được gửi lên.')
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "message": f"Lỗi: {response_json['message']}",
                                "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                            })
                            st.session_state.dashboard_info = None
                            st.session_state.logs = response_json.get('logs', 'Không có log nào được gửi lên.')
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "message": f"Lỗi: Không thể kết nối đến API. {str(e)}",
                            "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                        })
                        st.session_state.dashboard_info = None
                        st.session_state.logs = 'Lỗi khi kết nối API, không có log.'
                    st.rerun()
        with col2:
            if st.button("Xóa lịch sử", key="clear", type="secondary"):
                st.session_state.chat_history = []
                st.session_state.dashboard_info = None
                st.rerun()

    # Debug lịch sử chat (thêm để kiểm tra)
    st.write(f"Số tin nhắn trong lịch sử: {len(st.session_state.chat_history)}")
    st.write("Nội dung lịch sử chat:", st.session_state.chat_history)

# Hàm tạo dashboard
def create_dashboard(data, visualization):
    visualization_type = visualization.get("type", "none")
    required_columns = visualization.get("required_columns", [])
    
    if visualization_type == "table":
        # Kiểm tra dữ liệu trước khi tạo DataFrame
        if not data:
            st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu trống.</p>", unsafe_allow_html=True)
            return
        
        df = pd.DataFrame(data)
        # Kiểm tra các cột cần thiết có tồn tại không
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {', '.join(missing_columns)}.</p>", unsafe_allow_html=True)
            return
        
        st.dataframe(df[required_columns], use_container_width=True)
    elif visualization_type == "time series":
        try:
            # Chuẩn hóa dữ liệu
            if isinstance(data, dict) and "result" in data:
                data = data["result"]
            if not isinstance(data, list):
                st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu không phải danh sách hợp lệ.</p>", unsafe_allow_html=True)
                return
            
            # Kiểm tra dữ liệu
            if not data or not all(isinstance(item, dict) and "date" in item and "close_price" in item for item in data):
                st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu không chứa cột date hoặc close_price.</p>", unsafe_allow_html=True)
                return
            
            # Tạo DataFrame và vẽ biểu đồ
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            st.line_chart(df[["close_price"]], use_container_width=True)
        except Exception as e:
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ biểu đồ: {str(e)}</p>", unsafe_allow_html=True)

# Khu vực dashboard
with dashboard_col:
    # Hiển thị dashboard (nếu có)
    if st.session_state.dashboard_info and st.session_state.dashboard_info['enabled']:
        create_dashboard(st.session_state.dashboard_info['data'], st.session_state.dashboard_info['visualization'])
    else:
        st.markdown("<p style='text-align: center; color: #888;'>Không có dữ liệu để hiển thị.</p>", unsafe_allow_html=True)
    
    # Hiển thị log trong một expander
    st.markdown("### Log xử lý")
    with st.expander("Xem chi tiết log", expanded=False):
        st.text(st.session_state.get('logs', 'Chưa có log nào để hiển thị.'))