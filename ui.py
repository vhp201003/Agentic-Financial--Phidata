# ui.py
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
.thinking { color: #888; font-style: italic; }
</style>
""", unsafe_allow_html=True)

def load_visualization_metadata():
    """Load visualization metadata from visualization_metadata.yml."""
    vis_metadata_file = BASE_DIR / "config" / "visualization_metadata.yml"
    print(f"Loading visualization metadata from {vis_metadata_file}")
    try:
        with open(vis_metadata_file, "r") as file:
            vis_metadata = yaml.safe_load(file)
        logger.info("Successfully loaded visualization_metadata.yml")
        metadata = vis_metadata.get("visualization_metadata", [])
        supported_types = [entry["vis_type"].strip().lower() for entry in metadata]
        return supported_types
    except FileNotFoundError:
        logger.error("visualization_metadata.yml not found")
        return []
    except Exception as e:
        logger.error(f"Error loading visualization metadata: {str(e)}")
        return []
def process_stream_response(prompt, base_url):
    # Expander chỉ chứa thinking
    with st.expander("Quá trình suy nghĩ" if language == "vi" else "Thinking Process", expanded=True):
        thinking_container = st.empty()
        thinking_messages = []  # Danh sách thông điệp để lưu lịch sử

        assistant_message = {
            "role": "assistant",
            "content": "",
            "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            "visualizations": []
        }

        def stream_message(message):
            """Hàm hỗ trợ để stream từng thông điệp với st.write_stream."""
            for char in message:
                yield char
                time.sleep(0.02)  # Tốc độ gõ chữ
            yield "\n"  # Thêm xuống dòng sau mỗi thông điệp

        token_metrics = None  # Khởi tạo để tránh lỗi undefined
        try:
            response = requests.get(f"{base_url}/process_query?query={prompt}", stream=True, timeout=30)
            response.raise_for_status()

            current_event = None
            current_data = ""

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    logger.info(f"Received stream line: {decoded_line}")

                    if decoded_line.startswith("event:"):
                        current_event = decoded_line.split(":", 1)[1].strip()
                    elif decoded_line.startswith("data:"):
                        current_data = decoded_line[5:].strip()
                        try:
                            data = json.loads(current_data)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON: {e}, data: {current_data}")
                            continue

                        if current_event == "thinking":
                            message = data['message']
                            # Định dạng thông điệp
                            if message.startswith(("SQL:", "Kết quả SQL:", "Orchestrator:", "RAG:", "Chat Completion:", "Visualized:")):
                                formatted_message = f"```json\n{message}\n```"
                            else:
                                formatted_message = f"- {message}"
                            thinking_messages.append(formatted_message)
                            # Stream thông điệp mới và hiển thị lịch sử
                            thinking_container.markdown("\n".join(thinking_messages), unsafe_allow_html=True)
                            st.write_stream(stream_message(formatted_message))
                        elif current_event == "result":
                            thinking_container.empty()
                            response_json = data
                            assistant_message["content"] = response_json.get('message', 'Không có phản hồi chi tiết.')
                            if response_json.get('status') == 'success':
                                if 'data' in response_json and 'dashboard' in response_json['data'] and response_json['data']['dashboard']['enabled']:
                                    dashboard_info = response_json['data']['dashboard']
                                    create_dashboard(
                                        dashboard_info['data'],
                                        dashboard_info['visualization'],
                                        assistant_message["timestamp"],
                                        assistant_message["visualizations"]
                                    )
                                st.session_state.logs = response_json.get('logs', 'Không có log.')
                                token_metrics = response_json['data'].get('token_metrics')
                            else:
                                assistant_message["content"] = f"Lỗi: {response_json.get('message', 'Không có thông tin lỗi.')}"
                                st.session_state.logs = response_json.get('logs', 'Không có log.')
                            break
                        elif current_event == "error":
                            thinking_container.empty()
                            assistant_message["content"] = f"Lỗi: {data.get('message', 'Không có thông tin lỗi.')}"
                            st.session_state.logs = 'Lỗi khi xử lý stream.'
                            break

        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}")
            thinking_container.empty()
            assistant_message["content"] = f"Lỗi: Không thể kết nối đến API. {str(e)}" if language == "vi" else f"Error: Could not connect to API. {str(e)}"
            st.session_state.logs = 'Lỗi khi kết nối API, không có log.' if language == "vi" else 'Error connecting to API, no logs.'

    # Phản hồi cuối cùng nằm ngoài expander
    st.session_state.chat_history.append(assistant_message)
    with st.chat_message("assistant"):
        st.markdown(f"**Assistant** - {assistant_message['timestamp']}")
        st.markdown(assistant_message["content"], unsafe_allow_html=True)
        for vis in assistant_message["visualizations"]:
            st.plotly_chart(vis["fig"], key=vis["key"])

    # Hiển thị token metrics
    if token_metrics:
        with st.expander("Thông tin sử dụng Token" if language == "vi" else "Token Usage", expanded=False):
            st.markdown("**Chi tiết sử dụng Token:**")
            for agent, metrics in token_metrics.items():
                input_tokens = metrics['input_tokens'][0] if isinstance(metrics['input_tokens'], list) and metrics['input_tokens'] else metrics['input_tokens']
                output_tokens = metrics['output_tokens'][0] if isinstance(metrics['output_tokens'], list) and metrics['output_tokens'] else metrics['output_tokens']
                total_tokens_value = metrics['total_tokens'][0] if isinstance(metrics['total_tokens'], list) and metrics['total_tokens'] else metrics['total_tokens']
                st.markdown(f"- **{agent.replace('_', ' ').title()}:** Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens_value}")
            total_tokens_sum = sum(
                metrics['total_tokens'][0] if isinstance(metrics['total_tokens'], list) and metrics['total_tokens'] else metrics['total_tokens']
                for metrics in token_metrics.values()
            )
            st.markdown(
                f"<p style='text-align: center; color: #888;'>Tổng số token sử dụng: {total_tokens_sum}</p>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<p style='text-align: center; color: #888;'>Không có thông tin token.</p>",
            unsafe_allow_html=True
        )

def normalize_visualization_type(vis_type: str) -> str:
    """Normalize visualization type: replace spaces with underscores and convert to lowercase."""
    return vis_type.strip().lower().replace(" ", "_")

def create_dashboard(data, visualization, timestamp, vis_list):
    try:
        visualization_type = normalize_visualization_type(visualization.get("type", "none"))
        ui_requirements = visualization.get("ui_requirements", {})

        if not data:
            if st.session_state.get("show_debug", False):
                st.write("Debug: No data provided for dashboard")
            st.markdown("<p style='text-align: center; color: #888;'>No data available.</p>", unsafe_allow_html=True)
            return
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Invalid data format: {type(data)}")
            st.markdown("<p style='text-align: center; color: #888;'>Invalid data format (must be a list of dictionaries).</p>", unsafe_allow_html=True)
            return

        df = pd.DataFrame(data)
        if st.session_state.get("show_debug", False):
            st.write(f"Debug: DataFrame columns: {list(df.columns)}, Rows: {len(df)}")

        if df.empty or len(df.columns) == 0:
            if st.session_state.get("show_debug", False):
                st.write("Debug: DataFrame is empty or has no columns")
            st.markdown("<p style='text-align: center; color: #888;'>Empty data or no valid columns.</p>", unsafe_allow_html=True)
            return

        if visualization_type not in SUPPORTED_VISUALIZATION_TYPES:
            supported_types_str = ", ".join(SUPPORTED_VISUALIZATION_TYPES)
            logger.warning(f"Visualization type {visualization_type} not supported. Supported types: {supported_types_str}")
            st.markdown(f"<p style='text-align: center; color: #888;'>Unsupported chart type: {visualization_type}. Supported types: {supported_types_str}.</p>", unsafe_allow_html=True)
            return

        # 1. Table
        if visualization_type == "table":
            key = f"plotly_chart_table_{timestamp.replace(' ', '_').replace('/', '_')}_{len(vis_list)}"
            columns = ui_requirements.get("columns", list(df.columns))
            available_columns = [col for col in columns if col in df.columns]
            if not available_columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: No available columns for table. Required: {columns}, Available: {list(df.columns)}")
                st.markdown("<p style='text-align: center; color: #888;'>No valid columns available for display.</p>", unsafe_allow_html=True)
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
            x_col = ui_requirements.get("x_col")
            y_col = ui_requirements.get("y_col")
            additional_lines = ui_requirements.get("additional_lines", [])

            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing or invalid columns: x_col={x_col}, y_col={y_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Missing or invalid columns for line chart.</p>", unsafe_allow_html=True)
                return

            try:
                df[x_col] = pd.to_datetime(df[x_col])
            except Exception as e:
                logger.error(f"Error converting {x_col} to datetime: {str(e)}")
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Error converting {x_col} to datetime: {str(e)}")
                st.markdown("<p style='text-align: center; color: #888;'>Cannot format date column.</p>", unsafe_allow_html=True)
                return

            df = df.sort_values(x_col)

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering line chart with x_col={x_col}, y_col={y_col}, additional_lines={additional_lines}")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode="lines+markers",
                    name=y_col.replace('_', ' ').title(),
                    line=dict(color='rgb(0, 123, 255)'),
                    marker=dict(size=8)
                )
            )
            colors = ['rgb(255, 99, 71)', 'rgb(75, 192, 192)', 'rgb(255, 205, 86)']
            for idx, line_col in enumerate(additional_lines):
                if line_col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df[x_col],
                            y=df[line_col],
                            mode="lines",
                            name=line_col.replace('_', ' ').title(),
                            line=dict(color=colors[idx % len(colors)]),
                            marker=dict(size=8)
                        )
                    )
            fig.update_layout(
                title=f"Line Chart of {y_col.replace('_', ' ').title()} over {x_col.replace('_', ' ').title()}",
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=True,
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
            category_col = ui_requirements.get("category_col")
            value_col = ui_requirements.get("value_col")

            if not category_col or not value_col or category_col not in df.columns or value_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing or invalid columns: category_col={category_col}, value_col={value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Missing or invalid columns for bar chart.</p>", unsafe_allow_html=True)
                return

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering bar chart with category_col={category_col}, value_col={value_col}")
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
            category_col = ui_requirements.get("category_col")
            value_col = ui_requirements.get("value_col")

            if not category_col or not value_col or category_col not in df.columns or value_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing or invalid columns: category_col={category_col}, value_col={value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Missing or invalid columns for pie chart.</p>", unsafe_allow_html=True)
                return

            df_pie = df[[category_col, value_col]]

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering pie chart with category_col={category_col}, value_col={value_col}")
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=df_pie[category_col],
                        values=df_pie[value_col],
                        textinfo='percent+label',
                        hovertemplate='%{label}: %{value} (%{percent})',
                        marker=dict(colors=['rgb(0, 123, 255)', 'rgb(255, 99, 71)', 'rgb(75, 192, 192)', 'rgb(255, 205, 86)'])
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
            value_col = ui_requirements.get("value_col")

            if not value_col or value_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing or invalid column: value_col={value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Missing or invalid column for histogram.</p>", unsafe_allow_html=True)
                return

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering histogram with value_col={value_col}")
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
            group_col = ui_requirements.get("group_col")
            value_col = ui_requirements.get("value_col")
            group_transform = ui_requirements.get("group_transform", None)

            if not group_col or not value_col or group_col not in df.columns or value_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing or invalid columns: group_col={group_col}, value_col={value_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Missing or invalid columns for boxplot.</p>", unsafe_allow_html=True)
                return

            if group_transform == "to_month" and "date" in group_col.lower():
                try:
                    df['month'] = pd.to_datetime(df[group_col]).dt.strftime('%Y-%m')
                    group_col = 'month'
                except Exception as e:
                    logger.error(f"Error converting {group_col} to month: {str(e)}")
                    if st.session_state.get("show_debug", False):
                        st.write(f"Debug: Error converting {group_col} to month: {str(e)}")
                    st.markdown("<p style='text-align: center; color: #888;'>Cannot format date column.</p>", unsafe_allow_html=True)
                    return

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering boxplot with group_col={group_col}, value_col={value_col}")
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
                st.markdown("<p style='text-align: center; color: #888;'>No valid data for boxplot.</p>", unsafe_allow_html=True)
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
            x_col = ui_requirements.get("x_col")
            y_col = ui_requirements.get("y_col")
            label_col = ui_requirements.get("label_col")

            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                if st.session_state.get("show_debug", False):
                    st.write(f"Debug: Missing or invalid columns: x_col={x_col}, y_col={y_col}")
                st.markdown(f"<p style='text-align: center; color: #888;'>Missing or invalid columns for scatter plot.</p>", unsafe_allow_html=True)
                return

            if st.session_state.get("show_debug", False):
                st.write(f"Debug: Rendering scatter plot with x_col={x_col}, y_col={y_col}, label_col={label_col}")
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
            
            # Lấy danh sách tickers từ query hoặc metadata
            tickers = ui_requirements.get("tickers", [])
            if not tickers:
                st.markdown("<p style='text-align: center; color: #888;'>No tickers provided for heatmap.</p>", unsafe_allow_html=True)
                return

            # Tạo danh sách cột cho ma trận tương quan
            columns = [f"{t1.lower()}_{t2.lower()}" for i, t1 in enumerate(tickers) for t2 in tickers[i:]]
            
            if not all(col in df.columns for col in columns):
                st.write(f"Debug: Missing columns for heatmap: {columns}")
                st.markdown("<p style='text-align: center; color: #888;'>No valid columns for heatmap.</p>", unsafe_allow_html=True)
                return

            # Tạo ma trận tương quan
            matrix = []
            for t1 in tickers:
                row = []
                for t2 in tickers:
                    col_name = f"{t1.lower()}_{t2.lower()}" if t1 <= t2 else f"{t2.lower()}_{t1.lower()}"
                    value = df[col_name].iloc[0] if col_name in df.columns else df[f"{t2.lower()}_{t1.lower()}"].iloc[0]
                    row.append(value)
                matrix.append(row)
            matrix = np.array(matrix, dtype=float)

            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=tickers,
                    y=tickers,
                    colorscale=[[0, 'rgb(0, 123, 255)'], [1, 'rgb(255, 99, 71)']],
                    hoverongaps=False,
                    text=np.round(matrix, 2).astype(str),
                    texttemplate="%{text}",
                    textfont={"size": 12}
                )
            )
            fig.update_layout(
                title="Correlation Matrix of Daily Returns",
                xaxis_title="Tickers",
                yaxis_title="Tickers",
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="closest",
                xaxis=dict(tickangle=45)
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
        st.markdown(f"<p style='text-align: center; color: #888;'>Error rendering chart: {str(e)}. Please check the data or contact support.</p>", unsafe_allow_html=True)

SUPPORTED_VISUALIZATION_TYPES = load_visualization_metadata()

# with st.sidebar:
#     st.title('Financial Assistant Chatbot')
#     st.write('A financial analysis chatbot powered by API.')
#     st.subheader('Options')
#     language = st.selectbox("Language", ["vi", "en"], key="language")
#     show_debug = st.checkbox("Show Debug Info", value=False, key="show_debug")
#     if st.button('Clear Chat History', key="clear", type="secondary"):
#         st.session_state.chat_history = [{
#             "role": "assistant",
#             "content": "How can I assist you today?" if language == "en" else "Tôi có thể giúp gì cho bạn hôm nay?",
#             "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
#         }]
#         st.session_state.dashboard_info = None
#         st.session_state.logs = None
#         st.rerun()
language = "vi"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant",
        "content": "How can I assist you today?" if language == "en" else "Tôi có thể giúp gì cho bạn hôm nay?",
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
            st.markdown("<p style='text-align: center; color: #888;'>No content to display.</p>", unsafe_allow_html=True)
        
        if chat["role"] == "assistant":
            for vis in chat.get("visualizations", []):
                st.plotly_chart(vis["fig"], key=vis["key"])

# if st.session_state.logs:
#     with st.sidebar.expander("View detailed logs" if language == "en" else "Xem chi tiết log", expanded=False):
#         st.text(st.session_state.logs)

placeholder = "Enter your query (e.g., What is the closing price of Apple on 01/01/2025?)" if language == "en" else "Nhập câu hỏi của bạn (ví dụ: Giá đóng cửa của Apple ngày 01/01/2025?)"
if prompt := st.chat_input(placeholder):
    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    st.session_state.chat_history.append({"role": "user", "content": prompt, "timestamp": timestamp})
    
    with st.chat_message("user"):
        st.markdown(f"**User** - {timestamp}")
        st.write(prompt)

    base_url = API_URL.replace("/team", "")  # Đảm bảo dùng base URL đúng
    process_stream_response(prompt, base_url)