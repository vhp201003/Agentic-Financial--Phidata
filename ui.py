import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import markdown
from datetime import datetime
from config.ui_config import API_URL

# Cấu hình
st.set_page_config(layout="wide")
API_URL = API_URL + "/team"

# Palette màu đồng bộ
COLOR_PALETTE = {
    "primary": "#4CAF50",  # Xanh lá
    "secondary": "#007bff",  # Xanh dương
    "neutral": "#888",  # Xám
}

def markdown_table_to_html(markdown_text):
    import re
    table_pattern = r'\|.*?\|\n\|[-|:\s]+\|\n(?:\|.*?\|\n)*'
    tables = re.findall(table_pattern, markdown_text, re.MULTILINE)
    
    if not tables:
        return markdown.markdown(markdown_text)
    
    html_content = markdown_text
    for table in tables:
        rows = table.strip().split('\n')
        if len(rows) < 2:
            continue
            
        headers = [h.strip() for h in rows[0].split('|') if h.strip()]
        body_rows = [[cell.strip() for cell in row.split('|') if cell.strip()] for row in rows[2:]]
        
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
        
        html_content = html_content.replace(table, html_table)
    
    return markdown.markdown(html_content)

# Ánh xạ tên cột để xử lý sự không khớp
COLUMN_MAPPING = {
    "average volume": "avg_volume",
    "average close_price": "avg_close_price",
    "volume": "avg_volume",
    "close_price": "avg_close_price"
}

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
                    {markdown_table_to_html(chat['message'])}
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
                        print(response_data, flush=True)
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

    # Debug lịch sử chat
    st.write(f"Số tin nhắn trong lịch sử: {len(st.session_state.chat_history)}")
    st.write("Nội dung lịch sử chat:", st.session_state.chat_history)

# Hàm tạo dashboard
def create_dashboard(data, visualization):
    visualization_type = visualization.get("type", "none")
    required_columns = visualization.get("required_columns", [])
    aggregation = visualization.get("aggregation", None)
    
    # Kiểm tra dữ liệu đầu vào
    if not data:
        st.write("Debug: No data provided for dashboard")
        st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu trống.</p>", unsafe_allow_html=True)
        return

    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame(data)
    st.write(f"Debug: DataFrame columns: {list(df.columns)}, Rows: {len(df)}")
    
    # Ánh xạ required_columns sang tên cột thực tế
    mapped_columns = [COLUMN_MAPPING.get(col, col) for col in required_columns]
    
    # 1. Table
    if visualization_type == "table":
        available_columns = [col for col in mapped_columns if col in df.columns]
        missing_columns = [col for col in mapped_columns if col not in df.columns]
        
        if not available_columns:
            st.write(f"Debug: Missing columns: {missing_columns}")
            st.markdown("<p style='text-align: center; color: #888;'>Không có cột dữ liệu nào khả dụng để hiển thị.</p>", unsafe_allow_html=True)
            return
        
        st.write(f"Debug: Rendering table with columns: {available_columns}")
        fig = go.Figure(data=[go.Table(
            header=dict(values=available_columns, fill_color=COLOR_PALETTE["secondary"], align='center', font=dict(color='white')),
            cells=dict(values=[df[col].astype(str) for col in available_columns], fill_color=COLOR_PALETTE["neutral"], align='center')
        )])
        fig.update_layout(title="Data Table", margin=dict(l=10, r=10, t=30, b=10), height=400)
        st.plotly_chart(fig, use_container_width=True)

    # 2. Time Series
    elif visualization_type == "time series":
        try:
            if isinstance(data, dict) and "result" in data:
                data = data["result"]
                df = pd.DataFrame(data)
                st.write(f"Debug: Extracted 'result' from data, columns: {list(df.columns)}")
            
            if not isinstance(data, list):
                st.write("Debug: Data is not a valid list")
                st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu không phải danh sách hợp lệ.</p>", unsafe_allow_html=True)
                return
            
            if not data or not all(isinstance(item, dict) and "date" in item for item in data):
                st.write("Debug: Data missing 'date' column")
                st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu không chứa cột date.</p>", unsafe_allow_html=True)
                return

            value_col = next((col for col in ["close_price", "volume"] if col in df.columns), None)
            if not value_col:
                st.write("Debug: Data missing 'close_price' or 'volume' columns")
                st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu không chứa cột close_price hoặc volume.</p>", unsafe_allow_html=True)
                return
            
            st.write(f"Debug: Rendering time series with date and {value_col}")
            fig = px.line(df, x="date", y=value_col, title=f"Time Series of {value_col}",
                          color_discrete_sequence=[COLOR_PALETTE["secondary"]])
            fig.update_layout(
                xaxis_title="Date", yaxis_title=value_col, margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified", showlegend=False, xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write(f"Debug: Error in time series: {str(e)}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ biểu đồ: {str(e)}</p>", unsafe_allow_html=True)

    # 3. Bar Chart
    elif visualization_type == "bar":
        try:
            if len(mapped_columns) != 2:
                st.markdown("<p style='text-align: center; color: #888;'>Bar chart yêu cầu đúng 2 cột: danh mục và giá trị.</p>", unsafe_allow_html=True)
                return
            category_col, value_col = mapped_columns
            if category_col not in df.columns or value_col not in df.columns:
                st.write(f"Debug: Missing columns: {category_col} or {value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {category_col} hoặc {value_col}.</p>", unsafe_allow_html=True)
                return

            st.write(f"Debug: Rendering bar chart with {category_col} and {value_col}")
            fig = px.bar(df, x=category_col, y=value_col, title=f"{value_col} by {category_col}",
                         color_discrete_sequence=[COLOR_PALETTE["primary"]])
            fig.update_layout(
                xaxis_title=category_col, yaxis_title=value_col, margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified", showlegend=False, xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write(f"Debug: Error in bar chart: {str(e)}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ bar chart: {str(e)}</p>", unsafe_allow_html=True)

    # 4. Pie Chart
    elif visualization_type == "pie":
        try:
            if aggregation == "count":
                if len(mapped_columns) != 1:
                    st.markdown("<p style='text-align: center; color: #888;'>Pie chart với aggregation 'count' yêu cầu đúng 1 cột danh mục.</p>", unsafe_allow_html=True)
                    return
                category_col = mapped_columns[0]
                if category_col not in df.columns:
                    st.write(f"Debug: Missing column: {category_col}")
                    st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {category_col}.</p>", unsafe_allow_html=True)
                    return
                df_pie = df[category_col].value_counts().reset_index()
                df_pie.columns = [category_col, 'count']
                value_col = 'count'
            else:
                if len(mapped_columns) != 2:
                    st.markdown("<p style='text-align: center; color: #888;'>Pie chart yêu cầu đúng 2 cột: danh mục và giá trị.</p>", unsafe_allow_html=True)
                    return
                category_col, value_col = mapped_columns
                if category_col not in df.columns or value_col not in df.columns:
                    st.write(f"Debug: Missing columns: {category_col} or {value_col}")
                    st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {category_col} hoặc {value_col}.</p>", unsafe_allow_html=True)
                    return
                df_pie = df[[category_col, value_col]]

            st.write(f"Debug: Rendering pie chart with {category_col} and {value_col}")
            fig = px.pie(df_pie, names=category_col, values=value_col, title=f"Distribution of {value_col} by {category_col}",
                         color_discrete_sequence=[COLOR_PALETTE["primary"], COLOR_PALETTE["secondary"]])
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write(f"Debug: Error in pie chart: {str(e)}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ pie chart: {str(e)}</p>", unsafe_allow_html=True)

    # 5. Histogram
    elif visualization_type == "histogram":
        try:
            if not mapped_columns or len(mapped_columns) != 1:
                st.markdown("<p style='text-align: center; color: #888;'>Histogram yêu cầu đúng 1 cột dữ liệu.</p>", unsafe_allow_html=True)
                return
            col = mapped_columns[0]
            if col not in df.columns:
                st.write(f"Debug: Missing column: {col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {col}.</p>", unsafe_allow_html=True)
                return

            st.write(f"Debug: Rendering histogram with {col}")
            fig = px.histogram(df, x=col, title=f"Histogram of {col}", nbins=30,
                               color_discrete_sequence=[COLOR_PALETTE["primary"]])
            fig.update_layout(
                xaxis_title=col, yaxis_title="Frequency", margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified", showlegend=False, xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write(f"Debug: Error in histogram: {str(e)}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ histogram: {str(e)}</p>", unsafe_allow_html=True)

    # 6. Boxplot
    elif visualization_type == "boxplot":
        try:
            if len(mapped_columns) != 2:
                st.markdown("<p style='text-align: center; color: #888;'>Boxplot yêu cầu đúng 2 cột: nhóm và giá trị.</p>", unsafe_allow_html=True)
                return
            group_col, value_col = mapped_columns
            if group_col not in df.columns or value_col not in df.columns:
                st.write(f"Debug: Missing columns: {group_col} or {value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {group_col} hoặc {value_col}.</p>", unsafe_allow_html=True)
                return

            st.write(f"Debug: Rendering boxplot with {group_col} and {value_col}")
            fig = px.box(df, x=group_col, y=value_col, title=f"Boxplot of {value_col} grouped by {group_col}",
                         color_discrete_sequence=[COLOR_PALETTE["secondary"]])
            fig.update_layout(
                xaxis_title=group_col, yaxis_title=value_col, margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified", showlegend=False, xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write(f"Debug: Error in boxplot: {str(e)}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ boxplot: {str(e)}</p>", unsafe_allow_html=True)

    # 7. Scatter Plot
    elif visualization_type == "scatter":
        try:
            if len(mapped_columns) != 2:
                st.markdown("<p style='text-align: center; color: #888;'>Scatter plot yêu cầu đúng 2 cột: x và y.</p>", unsafe_allow_html=True)
                return
            x_col, y_col = mapped_columns
            if x_col not in df.columns or y_col not in df.columns:
                st.write(f"Debug: Missing columns: {x_col} or {y_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {x_col} hoặc {y_col}.</p>", unsafe_allow_html=True)
                return

            st.write(f"Debug: Rendering scatter plot with {x_col} and {y_col}")
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}",
                             color_discrete_sequence=[COLOR_PALETTE["primary"]])
            fig.update_layout(
                xaxis_title=x_col, yaxis_title=y_col, margin=dict(l=10, r=10, t=30, b=10),
                hovermode="closest", showlegend=False, xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write(f"Debug: Error in scatter plot: {str(e)}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ scatter plot: {str(e)}</p>", unsafe_allow_html=True)

    # 8. Heatmap
    elif visualization_type == "heatmap":
        try:
            if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
                st.write("Debug: Invalid data format for heatmap")
                st.markdown("<p style='text-align: center; color: #888;'>Heatmap yêu cầu dữ liệu dạng ma trận (list of lists).</p>", unsafe_allow_html=True)
                return

            matrix = np.array(data)
            st.write("Debug: Rendering heatmap")
            fig = px.imshow(matrix, title="Correlation Matrix",
                            color_continuous_scale=[COLOR_PALETTE["primary"], COLOR_PALETTE["secondary"]])
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write(f"Debug: Error in heatmap: {str(e)}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ heatmap: {str(e)}</p>", unsafe_allow_html=True)

    else:
        st.write(f"Debug: Unsupported visualization type: {visualization_type}")
        st.markdown(f"<p style='text-align: center; color: #888;'>Loại biểu đồ không được hỗ trợ: {visualization_type}.</p>", unsafe_allow_html=True)

# Khu vực dashboard
with dashboard_col:
    if st.session_state.dashboard_info and st.session_state.dashboard_info['enabled']:
        st.write("Debug: Dashboard enabled, rendering data...")
        create_dashboard(st.session_state.dashboard_info['data'], st.session_state.dashboard_info['visualization'])
    else:
        st.write("Debug: Dashboard not enabled or no data.")
        st.markdown("<p style='text-align: center; color: #888;'>Không có dữ liệu để hiển thị.</p>", unsafe_allow_html=True)
    
    st.markdown("### Log xử lý")
    with st.expander("Xem chi tiết log", expanded=False):
        st.text(st.session_state.get('logs', 'Chưa có log nào để hiển thị.'))