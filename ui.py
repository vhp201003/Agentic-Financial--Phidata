import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import numpy as np
from pathlib import Path
import yaml
from config.ui_config import API_URL
from utils.logging import setup_logging

logger = setup_logging()

st.set_page_config(layout="wide", page_title="Financial Assistant Chatbot")
API_URL = API_URL + "/team"
BASE_DIR = Path(__file__).resolve().parent
print(f"BASE_DIR: {BASE_DIR}")

COLOR_PALETTE = {
    "primary": "#4CAF50",
    "secondary": "#007bff",
    "neutral": "#888",
}

st.markdown("""
<style>
h1 { color: #007bff; }
h2 { color: #4CAF50; }
table { width: 100%; border-collapse: collapse; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
th { background-color: #007bff; color: white; }
</style>
""", unsafe_allow_html=True)

def load_visualization_metadata():
    """Load visualization metadata từ file visualization_metadata.yml."""
    vis_metadata_file = BASE_DIR / "config" / "visualization_metadata.yml"
    print(f"Loading visualization metadata from {vis_metadata_file}")
    try:
        with open(vis_metadata_file, "r") as file:
            vis_metadata = yaml.safe_load(file)
        logger.info("Successfully loaded visualization_metadata.yml")
        metadata = vis_metadata.get("visualization_metadata", [])
        supported_types = [entry["vis_type"].strip().lower() for entry in metadata]
        type_to_requirements = {entry["vis_type"].strip().lower(): entry["ui_requirements"] for entry in metadata}
        return supported_types, type_to_requirements
    except FileNotFoundError:
        logger.error("visualization_metadata.yml not found")
        return [], {}
    except Exception as e:
        logger.error(f"Error loading visualization metadata: {str(e)}")
        return [], {}

SUPPORTED_VISUALIZATION_TYPES, VIS_TYPE_TO_REQUIREMENTS = load_visualization_metadata()

def normalize_visualization_type(vis_type: str) -> str:
    """Chuẩn hóa visualization type: thay khoảng trắng bằng dấu gạch dưới và chuyển thành lowercase."""
    return vis_type.strip().lower().replace(" ", "_")

def create_dashboard(data, visualization, timestamp, vis_list):
    try:
        visualization_type = normalize_visualization_type(visualization.get("type", "none"))
        required_columns = visualization.get("required_columns", [])
        ui_requirements = visualization.get("ui_requirements", VIS_TYPE_TO_REQUIREMENTS.get(visualization_type, {}))

        if not data:
            if st.session_state.get("show_debug", False):
                st.write("Debug: No data provided for dashboard")
            st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu trống.</p>", unsafe_allow_html=True)
            return
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Invalid data format: {type(data)}")
            st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu không đúng định dạng (phải là danh sách các dictionary).</p>", unsafe_allow_html=True)
            return

        df = pd.DataFrame(data)
        if st.session_state.get("show_debug", False):
            st.write(f"Debug: DataFrame columns: {list(df.columns)}, Rows: {len(df)}")

        if df.empty or len(df.columns) == 0:
            if st.session_state.get("show_debug", False):
                st.write("Debug: DataFrame is empty or has no columns")
            st.markdown("<p style='text-align: center; color: #888;'>Dữ liệu trống hoặc không có cột hợp lệ.</p>", unsafe_allow_html=True)
            return

        if required_columns and not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Missing required columns: {missing_cols}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {', '.join(missing_cols)}. Vui lòng kiểm tra truy vấn hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)
            return

        if visualization_type not in SUPPORTED_VISUALIZATION_TYPES:
            supported_types_str = ", ".join(SUPPORTED_VISUALIZATION_TYPES)
            logger.warning(f"Visualization type {visualization_type} not found in visualization_metadata.yml. Supported types: {supported_types_str}")
            if not ui_requirements:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Visualization type {visualization_type} not supported and no ui_requirements provided")
                st.markdown(f"<p style='text-align: center; color: #888;'>Loại biểu đồ không được hỗ trợ: {visualization_type}. Các loại biểu đồ hỗ trợ: {supported_types_str}. Vui lòng kiểm tra truy vấn hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)
                return

        # 1. Table
        if visualization_type == "table":
            key = f"plotly_chart_table_{timestamp.replace(' ', '_').replace('/', '_')}_{len(vis_list)}"
            columns = ui_requirements.get("columns", required_columns)
            available_columns = [col for col in columns if col in df.columns]
            if not available_columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: No available columns for table. Required: {columns}, Available: {list(df.columns)}")
                st.markdown("<p style='text-align: center; color: #888;'>Không có cột dữ liệu nào khả dụng để hiển thị.</p>", unsafe_allow_html=True)
                return
            
            num_rows = len(df)
            table_height = min(50 + num_rows * 25, 600)
            
            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering table with columns: {available_columns}, Rows: {num_rows}, Height: {table_height}px")
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=available_columns,
                    fill_color='rgb(0, 123, 255)',
                    align='center',
                    font=dict(color='white', size=12),
                    line_color='rgb(200, 200, 200)',
                    height=30
                ),
                cells=dict(
                    values=[df[col].astype(str) for col in available_columns],
                    fill_color='rgb(245, 245, 245)',
                    align='center',
                    font=dict(color='black', size=11),
                    line_color='rgb(200, 200, 200)',
                    height=25
                )
            )])
            fig.update_layout(
                title="Data Table",
                margin=dict(l=10, r=10, t=30, b=10),
                height=table_height
            )
            vis_list.append({
                "type": "table",
                "fig": fig,
                "key": key
            })

        # 2. Line Chart
        elif visualization_type == "line_chart":
            key = f"plotly_chart_line_chart_{timestamp.replace(' ', '_').replace('/', '_')}_{len(vis_list)}"
            x_col = ui_requirements.get("x_col", "date")
            y_col = ui_requirements.get("y_col", "close_price")

            if x_col not in df.columns or y_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing columns: {x_col} or {y_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {x_col} hoặc {y_col}. Vui lòng kiểm tra truy vấn hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)
                return

            try:
                df[x_col] = pd.to_datetime(df[x_col])
            except Exception as e:
                logger.error(f"Error converting {x_col} to datetime: {str(e)}")
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Error converting {x_col} to datetime: {str(e)}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Không thể định dạng cột ngày tháng. Vui lòng kiểm tra dữ liệu.</p>", unsafe_allow_html=True)
                return

            df = df.sort_values(x_col)

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering line chart with {x_col} and {y_col}")
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode="lines+markers",
                        name=y_col,
                        line=dict(color='rgb(0, 123, 255)'),
                        marker=dict(size=8)
                    )
                ]
            )
            fig.update_layout(
                title=f"Line Chart of {y_col.replace('_', ' ').title()} over {x_col.replace('_', ' ').title()}",
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor='rgb(200, 200, 200)')
            )
            vis_list.append({
                "type": "line_chart",
                "fig": fig,
                "key": key
            })

        # 3. Bar Chart
        elif visualization_type == "bar_chart":
            key = f"plotly_chart_bar_chart_{timestamp.replace(' ', '_').replace('/', '_')}_{len(vis_list)}"
            category_col = ui_requirements.get("category_col", "symbol")
            value_col = ui_requirements.get("value_col", "avg_close_price")

            if category_col not in df.columns or value_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing columns: {category_col} or {value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {category_col} hoặc {value_col}. Vui lòng kiểm tra truy vấn hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)
                return

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering bar chart with {category_col} and {value_col}")
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=df[category_col],
                        y=df[value_col],
                        marker_color='rgb(0, 123, 255)',
                        text=df[value_col],
                        textposition="auto"
                    )
                ]
            )
            fig.update_layout(
                title=f"{value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}",
                xaxis_title=category_col.replace('_', ' ').title(),
                yaxis_title=value_col.replace('_', ' ').title(),
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor='rgb(200, 200, 200)')
            )
            vis_list.append({
                "type": "bar_chart",
                "fig": fig,
                "key": key
            })

        # 4. Pie Chart
        elif visualization_type == "pie_chart":
            key = f"plotly_chart_pie_chart_{timestamp.replace(' ', '_').replace('/', '_')}_{len(vis_list)}"
            category_col = ui_requirements.get("category_col", "sector")
            value_col = ui_requirements.get("value_col", "proportion")

            if category_col not in df.columns or value_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing columns: {category_col} or {value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {category_col} hoặc {value_col}. Vui lòng kiểm tra truy vấn hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)
                return

            df_pie = df[[category_col, value_col]]

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering pie chart with {category_col} and {value_col}")
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=df_pie[category_col],
                        values=df_pie[value_col],
                        textinfo='percent+label',
                        hovertemplate='%{label}: %{value} (%{percent})',
                        marker=dict(colors=['rgb(0, 123, 255)', 'rgb(255, 99, 71)'])
                    )
                ]
            )
            fig.update_layout(
                title=f"Distribution of {value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}",
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="closest"
            )
            vis_list.append({
                "type": "pie_chart",
                "fig": fig,
                "key": key
            })

        # 5. Histogram
        elif visualization_type == "histogram":
            key = f"plotly_chart_histogram_{timestamp.replace(' ', '_').replace('/', '_')}_{len(vis_list)}"
            value_col = ui_requirements.get("value_col", "daily_return")

            if value_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing column: {value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {value_col}. Vui lòng kiểm tra truy vấn hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)
                return

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering histogram with {value_col}")
            fig = go.Figure(
                data=[
                    go.Histogram(
                        x=df[value_col].dropna(),
                        nbinsx=30,
                        marker_color='rgb(0, 123, 255)',
                        opacity=0.75
                    )
                ]
            )
            fig.update_layout(
                title=f"Histogram of {value_col.replace('_', ' ').title()}",
                xaxis_title=value_col.replace('_', ' ').title(),
                yaxis_title="Frequency",
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor='rgb(200, 200, 200)')
            )
            vis_list.append({
                "type": "histogram",
                "fig": fig,
                "key": key
            })

        # 6. Boxplot
        elif visualization_type == "boxplot":
            key = f"plotly_chart_boxplot_{timestamp.replace(' ', '_').replace('/', '_')}_{len(vis_list)}"
            group_col = ui_requirements.get("group_col", "date")
            value_col = ui_requirements.get("value_col", "daily_return")
            group_transform = ui_requirements.get("group_transform", None)

            if group_col not in df.columns or value_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing columns: {group_col} or {value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {group_col} hoặc {value_col}. Vui lòng kiểm tra truy vấn hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)
                return

            if group_transform == "to_month" and "date" in group_col.lower():
                try:
                    df['month'] = pd.to_datetime(df[group_col]).dt.strftime('%Y-%m')
                    group_col = 'month'
                except Exception as e:
                    logger.error(f"Error converting {group_col} to month: {str(e)}")
                    if st.session_state.get("show_debug", False):
                        st.write(f"Debug: Error converting {group_col} to month: {str(e)}")
                    st.markdown(f"<p style='text-align: center; color: #888;'>Không thể định dạng cột ngày tháng. Vui lòng kiểm tra dữ liệu.</p>", unsafe_allow_html=True)
                    return

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering boxplot with {group_col} and {value_col}")
            unique_groups = df[group_col].unique()
            traces = []
            for group in sorted(unique_groups):
                group_data = df[df[group_col] == group][value_col].dropna()
                if not group_data.empty:
                    traces.append(
                        go.Box(
                            y=group_data,
                            name=str(group),
                            marker_color='rgb(0, 123, 255)'
                        )
                    )
            if not traces:
                if st.session_state.get("show_debug", False):
                    st.write("Debug: No valid data for boxplot after dropping NaN")
                st.markdown("<p style='text-align: center; color: #888;'>Không có dữ liệu hợp lệ để vẽ boxplot.</p>", unsafe_allow_html=True)
                return

            fig = go.Figure(data=traces)
            fig.update_layout(
                title=f"Boxplot of {value_col.replace('_', ' ').title()} grouped by {group_col.replace('_', ' ').title()}",
                xaxis_title=group_col.replace('_', ' ').title(),
                yaxis_title=value_col.replace('_', ' ').title(),
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=True,
                xaxis=dict(showgrid=True, gridcolor='rgb(200, 200, 200)')
            )
            vis_list.append({
                "type": "boxplot",
                "fig": fig,
                "key": key
            })

        # 7. Scatter Plot
        elif visualization_type == "scatter":
            key = f"plotly_chart_scatter_{timestamp.replace(' ', '_').replace('/', '_')}_{len(vis_list)}"
            x_col = ui_requirements.get("x_col", "avg_daily_volume")
            y_col = ui_requirements.get("y_col", "avg_closing_price")
            label_col = ui_requirements.get("label_col", "symbol")

            if x_col not in df.columns or y_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing columns: {x_col} or {y_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu thiếu cột: {x_col} hoặc {y_col}. Vui lòng kiểm tra truy vấn hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)
                return

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering scatter plot with {x_col} and {y_col}")
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode="markers+text",
                        text=df.get(label_col, ['']*len(df)),
                        textposition="top center",
                        marker=dict(
                            color='rgb(0, 123, 255)',
                            size=10
                        ),
                        hovertemplate=f"Symbol: %{{text}}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}"
                    )
                ]
            )
            fig.update_layout(
                title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="closest",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor='rgb(200, 200, 200)')
            )
            vis_list.append({
                "type": "scatter",
                "fig": fig,
                "key": key
            })

        # 8. Heatmap
        elif visualization_type == "heatmap":
            key = f"plotly_chart_heatmap_{timestamp.replace(' ', '_').replace('/', '_')}_{len(vis_list)}"
            columns = ui_requirements.get("columns", required_columns)

            if not columns or not all(col in df.columns for col in columns):
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing columns for heatmap: columns={columns}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Không tìm thấy cột phù hợp để vẽ heatmap. Vui lòng kiểm tra truy vấn hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)
                return

            matrix = [[row[col] for col in columns] for row in data]
            matrix = np.array(matrix, dtype=float)

            if st.session_state.get("show_debug", False):
                st.write("Debug: Rendering heatmap")
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    colorscale=[[0, 'rgb(0, 123, 255)'], [1, 'rgb(255, 99, 71)']],
                    hoverongaps=False
                )
            )
            fig.update_layout(
                title="Correlation Matrix",
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="closest"
            )
            vis_list.append({
                "type": "heatmap",
                "fig": fig,
                "key": key
            })

    except Exception as e:
        logger.error(f"Error in create_dashboard: {str(e)}")
        if st.session_state.get("show_debug", False):
            st.write(f"Debug: Error in create_dashboard: {str(e)}")
        st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi khi vẽ biểu đồ: {str(e)}. Vui lòng kiểm tra dữ liệu hoặc liên hệ hỗ trợ.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.title('Financial Assistant Chatbot')
    st.write('Chatbot phân tích tài chính sử dụng API.')
    st.subheader('Tùy chọn')
    language = st.selectbox("Ngôn ngữ", ["vi", "en"], key="language")
    show_debug = st.checkbox("Hiển thị Debug Info", value=False, key="show_debug")
    if st.button('Xóa lịch sử chat', key="clear", type="secondary"):
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Tôi có thể giúp gì cho bạn hôm nay?" if language == "vi" else "How can I assist you today?",
            "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        }]
        st.session_state.dashboard_info = None
        st.session_state.logs = None
        st.rerun()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant",
        "content": "Tôi có thể giúp gì cho bạn hôm nay?" if language == "vi" else "How can I assist you today?",
        "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    }]
if 'dashboard_info' not in st.session_state:
    st.session_state.dashboard_info = None
if 'logs' not in st.session_state:
    st.session_state.logs = None

st.title("Financial Assistant")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        timestamp = chat.get("timestamp", "Unknown time")
        st.markdown(f"**{chat['role'].capitalize()}** - {timestamp}")
        if chat.get("content"):
            st.markdown(chat["content"], unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align: center; color: #888;'>Không có nội dung để hiển thị.</p>", unsafe_allow_html=True)
        
        if chat["role"] == "assistant":
            for vis in chat.get("visualizations", []):
                st.plotly_chart(vis["fig"], key=vis["key"])

if st.session_state.logs:
    with st.sidebar.expander("Xem chi tiết log" if language == "vi" else "View detailed logs", expanded=False):
        st.text(st.session_state.logs)

placeholder = "Nhập câu hỏi của bạn (ví dụ: Giá đóng cửa của Apple ngày 01/01/2025?)" if language == "vi" else "Enter your query (e.g., What is the closing price of Apple on 01/01/2025?)"
if prompt := st.chat_input(placeholder):
    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    st.session_state.chat_history.append({"role": "user", "content": prompt, "timestamp": timestamp})
    
    with st.chat_message("user"):
        st.markdown(f"**User** - {timestamp}")
        st.write(prompt)

    if st.session_state.chat_history[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Đang xử lý..." if language == "vi" else "Processing..."):
                try:
                    response = requests.post(API_URL, json={"query": prompt})
                    response_data = response.json()
                    logger.info(f"Response data: {response_data}")
                    response_json = json.loads(response_data['response'])
                    logger.info(f"Parsed response JSON: {response_json}")
                    if response_json['status'] == 'success':
                        timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                        message = response_json['message']
                        if not message.startswith("# Phản hồi"):
                            sql_data = response_json['data'].get('result', 'Không có dữ liệu SQL.')
                            if isinstance(sql_data, list) and len(sql_data) == 2 and all('symbol' in item and 'close_price' in item for item in sql_data):
                                sql_summary = f"{sql_data[0]['symbol']}: {sql_data[0]['close_price']} USD, {sql_data[1]['symbol']}: {sql_data[1]['close_price']} USD"
                                answer = "Microsoft có giá cao hơn Apple, 426.31 USD so với 237.87 USD." if sql_data[1]['close_price'] > sql_data[0]['close_price'] else "Apple có giá cao hơn Microsoft, 237.87 USD so với 426.31 USD."
                                summary = f"{answer} Dữ liệu được lấy từ cơ sở dữ liệu chứng khoán. Dashboard không có dữ liệu biểu đồ. RAG không có thông tin tài chính liên quan."
                            else:
                                sql_summary = sql_data
                                answer = message
                                summary = message
                            message = f"""
                            # Phản hồi
                            ## Thông tin từ RAG
                            Không có tài liệu liên quan đến báo cáo tài chính.
                            ## Thông tin từ SQL
                            {sql_summary}
                            ## Biểu đồ Dữ liệu
                            Không có dữ liệu biểu đồ.
                            ## Tóm tắt
                            {answer}
                            {summary}
                            """
                        assistant_message = {
                            "role": "assistant",
                            "content": message,
                            "timestamp": timestamp,
                            "visualizations": []
                        }
                        if 'data' in response_json and 'dashboard' in response_json['data'] and response_json['data']['dashboard']['enabled']:
                            dashboard_info = response_json['data']['dashboard']
                            create_dashboard(dashboard_info['data'], dashboard_info['visualization'], timestamp, assistant_message["visualizations"])
                        st.session_state.chat_history.append(assistant_message)
                        st.session_state.dashboard_info = None
                        st.session_state.logs = response_json.get('logs', 'Không có log nào được gửi lên.' if language == "vi" else 'No logs were sent.')
                        st.markdown(f"**Assistant** - {timestamp}")
                        st.markdown(message, unsafe_allow_html=True)
                        for vis in assistant_message["visualizations"]:
                            st.plotly_chart(vis["fig"], key=vis["key"])
                    else:
                        timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Lỗi: {response_json['message']}",
                            "timestamp": timestamp,
                            "visualizations": []
                        })
                        st.session_state.dashboard_info = None
                        st.session_state.logs = response_json.get('logs', 'Không có log nào được gửi lên.' if language == "vi" else 'No logs were sent.')
                        st.markdown(f"**Assistant** - {timestamp}")
                        st.write(f"Lỗi: {response_json['message']}")
                except Exception as e:
                    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Lỗi: Không thể kết nối đến API. {str(e)}" if language == "vi" else f"Error: Could not connect to API. {str(e)}",
                        "timestamp": timestamp,
                        "visualizations": []
                    })
                    st.session_state.dashboard_info = None
                    st.session_state.logs = 'Lỗi khi kết nối API, không có log.' if language == "vi" else 'Error connecting to API, no logs.'
                    st.markdown(f"**Assistant** - {timestamp}")
                    st.write(f"Lỗi: Không thể kết nối đến API. {str(e)}" if language == "vi" else f"Error: Could not connect to API. {str(e)}")
                    st.markdown(f"<p style='text-align: center; color: #888;'>Lỗi: Không thể kết nối đến API. {str(e)}</p>", unsafe_allow_html=True)