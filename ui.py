import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import markdown
from datetime import datetime
import time
import numpy as np
from config.ui_config import API_URL

# Cấu hình
st.set_page_config(layout="wide", page_title="Financial Assistant Chatbot")
API_URL = API_URL + "/team"

# Palette màu đồng bộ
COLOR_PALETTE = {
    "primary": "#4CAF50",  # Xanh lá
    "secondary": "#007bff",  # Xanh dương
    "neutral": "#888",  # Xám
}

# Hàm chuyển bảng Markdown thành HTML
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

# Hàm tạo dashboard sử dụng plotly.graph_objects
def create_dashboard(data, visualization):
    import streamlit as st
    import plotly.graph_objects as go
    import time

    key = f"plotly_chart_{visualization.get('type', 'none')}_{int(time.time())}"
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
        
        num_rows = len(df)
        table_height = min(50 + num_rows * 25, 600)
        
        st.write(f"Debug: Rendering table with columns: {available_columns}, Rows: {num_rows}, Height: {table_height}px")
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=available_columns,
                fill_color=COLOR_PALETTE["secondary"],
                align='center',
                font=dict(color='white', size=12),
                line_color=COLOR_PALETTE["neutral"],
                height=30
            ),
            cells=dict(
                values=[df[col].astype(str) for col in available_columns],
                fill_color=COLOR_PALETTE["neutral"],
                align='center',
                font=dict(color='black', size=11),
                line_color=COLOR_PALETTE["neutral"],
                height=25
            )
        )])
        fig.update_layout(
            title="Data Table",
            margin=dict(l=10, r=10, t=30, b=10),
            height=table_height
        )
        st.plotly_chart(fig, key=key)

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
            
            dates = df["date"].tolist()
            values = df[value_col].tolist()

            st.write(f"Debug: Rendering time series with date and {value_col}")
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode="lines+markers",
                        name=value_col,
                        line=dict(color=COLOR_PALETTE["secondary"]),
                        marker=dict(size=8)
                    )
                ]
            )
            fig.update_layout(
                title=f"Time Series of {value_col}",
                xaxis_title="Date",
                yaxis_title=value_col,
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, key=key)
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
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=df[category_col],
                        y=df[value_col],
                        marker_color=COLOR_PALETTE["primary"],
                        text=df[value_col],
                        textposition="auto"
                    )
                ]
            )
            fig.update_layout(
                title=f"{value_col} by {category_col}",
                xaxis_title=category_col,
                yaxis_title=value_col,
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, key=key)
        except Exception as e:
            st.write(f"Debug: Error in bar chart: {str(e)}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ bar chart: {str(e)}</p>", unsafe_allow_html=True)

    # 4. Pie Chart
    elif visualization_type == "pie":
        try:
            if aggregation == "count":
                # Lấy cột danh mục (category_col) từ mapped_columns
                category_col = mapped_columns[0]
                if category_col not in df.columns:
                    st.write(f"Debug: Missing column: {category_col}")
                    st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {category_col}.</p>", unsafe_allow_html=True)
                    return
                
                # Kiểm tra xem dữ liệu đã có cột 'count' chưa
                if "count" in df.columns:
                    # Nếu đã có cột 'count', sử dụng cột này làm value_col
                    df_pie = df[[category_col, "count"]]
                    value_col = "count"
                else:
                    # Nếu không có cột 'count', tự tính count từ cột danh mục
                    df_pie = df[category_col].value_counts().reset_index()
                    df_pie.columns = [category_col, 'count']
                    value_col = 'count'
            else:
                # Trường hợp aggregation != "count" (ví dụ: sum, avg)
                if len(mapped_columns) != 2:
                    st.markdown("<p style='text-align: center; color: #888;'>Pie chart yêu cầu đúng 2 cột: danh mục và giá trị.</p>", unsafe_allow_html=True)
                    return
                category_col, value_col = mapped_columns
                if category_col not in df.columns or value_col not in df.columns:
                    st.write(f"Debug: Missing columns: {category_col} or {value_col}")
                    st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {category_col} hoặc {value_col}.</p>", unsafe_allow_html=True)
                    return
                df_pie = df[[category_col, value_col]]

            st.write(f"Debug: Rendering pie chart with data: {df_pie.to_dict()}")
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=df_pie[category_col],
                        values=df_pie[value_col],
                        textinfo='percent+label',
                        hovertemplate='%{label}: %{value} (%{percent})',
                        marker=dict(colors=[COLOR_PALETTE["primary"], COLOR_PALETTE["secondary"]])
                    )
                ]
            )
            fig.update_layout(
                title=f"Distribution of {value_col} by {category_col}",
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="closest"
            )
            st.plotly_chart(fig, key=key)
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
            fig = go.Figure(
                data=[
                    go.Histogram(
                        x=df[col],
                        nbinsx=30,
                        marker_color=COLOR_PALETTE["primary"],
                        opacity=0.75
                    )
                ]
            )
            fig.update_layout(
                title=f"Histogram of {col}",
                xaxis_title=col,
                yaxis_title="Frequency",
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, key=key)
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
            unique_groups = df[group_col].unique()
            traces = []
            for group in unique_groups:
                group_data = df[df[group_col] == group][value_col]
                traces.append(
                    go.Box(
                        y=group_data,
                        name=str(group),
                        marker_color=COLOR_PALETTE["secondary"]
                    )
                )
            fig = go.Figure(data=traces)
            fig.update_layout(
                title=f"Boxplot of {value_col} grouped by {group_col}",
                xaxis_title=group_col,
                yaxis_title=value_col,
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=True,
                xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, key=key)
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
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode="markers",
                        marker=dict(
                            color=COLOR_PALETTE["primary"],
                            size=10
                        ),
                        hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}"
                    )
                ]
            )
            fig.update_layout(
                title=f"{x_col} vs {y_col}",
                xaxis_title=x_col,
                yaxis_title=y_col,
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="closest",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["neutral"])
            )
            st.plotly_chart(fig, key=key)
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
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    colorscale=[[0, COLOR_PALETTE["primary"]], [1, COLOR_PALETTE["secondary"]]],
                    hoverongaps=False
                )
            )
            fig.update_layout(
                title="Correlation Matrix",
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="closest"
            )
            st.plotly_chart(fig, key=key)
        except Exception as e:
            st.write(f"Debug: Error in heatmap: {str(e)}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ heatmap: {str(e)}</p>", unsafe_allow_html=True)

    else:
        st.write(f"Debug: Unsupported visualization type: {visualization_type}")
        st.markdown(f"<p style='text-align: center; color: #888;'>Loại biểu đồ không được hỗ trợ: {visualization_type}.</p>", unsafe_allow_html=True)

# Ánh xạ tên cột để xử lý sự không khớp
COLUMN_MAPPING = {
    "average volume": "avg_volume",
    "average close_price": "avg_close_price",
    "volume": "avg_volume",
    "close_price": "avg_close_price"
}

# Sidebar
with st.sidebar:
    st.title('Financial Assistant Chatbot')
    st.write('Chatbot phân tích tài chính sử dụng API.')
    st.subheader('Tùy chọn')
    if st.button('Xóa lịch sử chat', key="clear", type="secondary"):
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Tôi có thể giúp gì cho bạn hôm nay?",
            "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        }]
        st.session_state.dashboard_info = None
        st.session_state.logs = None
        st.rerun()

# Khởi tạo session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant",
        "content": "Tôi có thể giúp gì cho bạn hôm nay?",
        "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    }]
if 'dashboard_info' not in st.session_state:
    st.session_state.dashboard_info = None
if 'logs' not in st.session_state:
    st.session_state.logs = None

# Tiêu đề
st.title("Financial Assistant")

# Hiển thị lịch sử chat
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        # Sử dụng get() để tránh lỗi KeyError, mặc định là "Unknown time"
        timestamp = chat.get("timestamp", "Unknown time")
        st.markdown(f"**{chat['role'].capitalize()}** - {timestamp}")
        st.markdown(markdown_table_to_html(chat["content"]), unsafe_allow_html=True)
        # Hiển thị dashboard nếu có
        if chat["role"] == "assistant" and st.session_state.dashboard_info and st.session_state.dashboard_info['enabled']:
            with st.container():
                create_dashboard(st.session_state.dashboard_info['data'], st.session_state.dashboard_info['visualization'])

# Hiển thị log nếu có
if st.session_state.logs:
    with st.sidebar.expander("Xem chi tiết log", expanded=False):
        st.text(st.session_state.logs)

# Nhập tin nhắn người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn (ví dụ: Giá đóng cửa của Apple ngày 01/01/2025?)"):
    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    st.session_state.chat_history.append({"role": "user", "content": prompt, "timestamp": timestamp})
    
    with st.chat_message("user"):
        st.markdown(f"**User** - {timestamp}")
        st.write(prompt)

    # Gửi yêu cầu đến API nếu tin nhắn cuối không phải từ assistant
    if st.session_state.chat_history[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Đang xử lý..."):
                try:
                    response = requests.post(API_URL, json={"query": prompt})
                    response_data = response.json()
                    response_json = json.loads(response_data['response'])
                    if response_json['status'] == 'success':
                        timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_json['message'],
                            "timestamp": timestamp
                        })
                        if 'data' in response_json and 'dashboard' in response_json['data'] and response_json['data']['dashboard']['enabled']:
                            st.session_state.dashboard_info = response_json['data']['dashboard']
                        else:
                            st.session_state.dashboard_info = None
                        st.session_state.logs = response_json.get('logs', 'Không có log nào được gửi lên.')
                        st.markdown(f"**Assistant** - {timestamp}")
                        st.markdown(markdown_table_to_html(response_json['message']), unsafe_allow_html=True)
                        # Hiển thị dashboard nếu có
                        if st.session_state.dashboard_info and st.session_state.dashboard_info['enabled']:
                            with st.container():
                                create_dashboard(st.session_state.dashboard_info['data'], st.session_state.dashboard_info['visualization'])
                    else:
                        timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Lỗi: {response_json['message']}",
                            "timestamp": timestamp
                        })
                        st.session_state.dashboard_info = None
                        st.session_state.logs = response_json.get('logs', 'Không có log nào được gửi lên.')
                        st.markdown(f"**Assistant** - {timestamp}")
                        st.write(f"Lỗi: {response_json['message']}")
                except Exception as e:
                    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Lỗi: Không thể kết nối đến API. {str(e)}",
                        "timestamp": timestamp
                    })
                    st.session_state.dashboard_info = None
                    st.session_state.logs = 'Lỗi khi kết nối API, không có log.'
                    st.markdown(f"**Assistant** - {timestamp}")
                    st.write(f"Lỗi: Không thể kết nối đến API. {str(e)}")

# # Debug lịch sử chat
# st.write(f"Số tin nhắn trong lịch sử: {len(st.session_state.chat_history)}")
# st.write("Nội dung lịch sử chat:", st.session_state.chat_history)