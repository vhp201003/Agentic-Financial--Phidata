import json
import re
import yaml
from pathlib import Path
from utils.logging import setup_logging
from phi.agent import RunResponse
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
logger = setup_logging()

def load_config() -> dict:
    config_file = Path.joinpath(BASE_DIR, "config/chat_completion_config.yml")
    print(f"Loading chat completion config from {config_file}")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    logger.info("Successfully loaded chat completion config")
    return config

def prepare_rag_summary(rag_documents: list, config: dict) -> str:
    if not rag_documents or not all(isinstance(doc, dict) and 'document' in doc and 'filename' in doc and 'company' in doc for doc in rag_documents):
        return config['formatting']['rag']['empty_message']['vi']

    rag_by_company = {}
    for doc in rag_documents:
        company = doc['company']
        if company not in rag_by_company:
            rag_by_company[company] = []
        content = doc['document'][:200] + ("..." if len(doc['document']) > 200 else "")
        rag_by_company[company].append(f"{company}: {content} from {doc['filename']}")
    return "\n".join(f"{company}: " + "; ".join(entries) for company, entries in rag_by_company.items())

def prepare_dashboard_summary(dashboard_info: dict, config: dict) -> str:
    if not dashboard_info.get('enabled', False) or not isinstance(dashboard_info.get('data', []), list) or len(dashboard_info['data']) == 0:
        return config['formatting']['dashboard']['empty_message']['vi']

    vis_type = dashboard_info['visualization'].get('type', 'none')
    ui_requirements = dashboard_info['visualization'].get('ui_requirements', {})
    template = config['formatting']['dashboard'].get('vis_type_templates', {}).get(vis_type, config['formatting']['dashboard'].get('default_template', {'vi': "Biểu đồ {vis_type} thể hiện dữ liệu."}))
    
    summary = template['vi'].format(
        vis_type=vis_type,
        group_col=ui_requirements.get('group_col', 'group'),
        value_col=ui_requirements.get('value_col', 'value'),
        x_col=ui_requirements.get('x_col', 'x'),
        y_col=ui_requirements.get('y_col', 'y'),
        category_col=ui_requirements.get('category_col', 'category')
    )

    if dashboard_info['data']:
        key_points = []
        for record in dashboard_info['data'][:3]:
            if 'sector' in record and 'proportion' in record:
                key_points.append(f"{record['sector']} ({record['proportion']}%)")
        if key_points:
            summary += " " + ", ".join(key_points) + "."
    return summary

def prepare_sql_summary(sql_response: str, config: dict, tickers: list, required_columns: list = None, dashboard_enabled: bool = False) -> str:
    if "Dữ liệu từ cơ sở dữ liệu" not in sql_response:
        return config['formatting']['sql']['empty_message']['vi']

    match = re.search(r'\[(.*)\]', sql_response, re.DOTALL)
    if not match:
        return config['formatting']['sql']['empty_message']['vi']

    try:
        json_str = match.group(1)
        data = json.loads(f"[{json_str}]")
        if not data:
            return config['formatting']['sql']['empty_message']['vi']

        summaries = []
        required_columns = required_columns or []
        if dashboard_enabled and required_columns and all(col in data[0] for col in required_columns):
            if "symbol" in required_columns and "close_price" in required_columns:
                ticker_map = {record['symbol']: record['close_price'] for record in data}
                formatted_data = [f"{ticker}: {ticker_map[ticker]} USD" for ticker in tickers if ticker in ticker_map]
                summaries.append(", ".join(formatted_data))
            elif "avg_close_price" in required_columns:
                formatted_data = [f"{tickers[0] if tickers else 'Company'}: {record['avg_close_price']} USD" for record in data]
                summaries.append(", ".join(formatted_data))
            elif "daily_return" in required_columns:
                valid_returns = [record['daily_return'] for record in data if isinstance(record['daily_return'], (int, float)) and not pd.isna(record['daily_return'])]
                if valid_returns:
                    avg_return = sum(valid_returns) / len(valid_returns)
                    summaries.append(f"{tickers[0] if tickers else 'Company'} Daily Returns: Trung bình {avg_return:.4f}")
                else:
                    summaries.append("Không có dữ liệu lợi nhuận hàng ngày hợp lệ.")
            elif "sector" in required_columns and "count" in required_columns:
                formatted_data = [f"{record['sector']}: {record['count']}" for record in data]
                summaries.append(", ".join(formatted_data))
            elif "date" in required_columns and "close_price" in required_columns:
                df = pd.DataFrame(data)
                df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
                monthly_avg = df.groupby('month')['close_price'].mean().round(2)
                formatted_data = [f"{month}: {price} USD" for month, price in monthly_avg.items()]
                summaries.append(", ".join(formatted_data))
            elif "avg_daily_volume" in required_columns and "avg_closing_price" in required_columns:
                formatted_data = [f"{record['symbol']}: Volume {record['avg_daily_volume']:.0f}, Price {record['avg_closing_price']:.2f} USD" for record in data[:5]]
                summaries.append(", ".join(formatted_data) + (", ..." if len(data) > 5 else ""))
        else:
            # Khi Dashboard: false, không validate required_columns, chỉ hiển thị dữ liệu thô
            for record in data:
                for key, value in record.items():
                    summaries.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(summaries)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse SQL data: {str(e)}")
        return config['formatting']['sql']['empty_message']['vi']

def chat_completion_flow(query: str, rag_documents: list, sql_response: str, dashboard_info: dict, chat_completion_agent, tickers: list = None) -> dict:
    try:
        if not isinstance(query, str):
            logger.error(f"Invalid query format: {type(query)}")
            raise ValueError("Query must be a string")
        if not isinstance(rag_documents, list):
            logger.error(f"Invalid RAG documents format: {type(rag_documents)}")
            raise ValueError("RAG documents must be a list")
        if not isinstance(sql_response, str):
            logger.error(f"Invalid SQL response format: {type(sql_response)}")
            raise ValueError("SQL response must be a string")
        if not isinstance(dashboard_info, dict):
            logger.error(f"Invalid dashboard info format: {type(dashboard_info)}")
            raise ValueError("Dashboard info must be a dict")

        config = load_config()

        rag_summary = prepare_rag_summary(rag_documents, config)
        sql_summary = prepare_sql_summary(sql_response, config, tickers or [], dashboard_info.get('visualization', {}).get('required_columns', []), dashboard_info.get('enabled', False))
        dashboard_summary = prepare_dashboard_summary(dashboard_info, config)

        # Kiểm tra xem có dữ liệu thực sự không
        has_rag_data = rag_summary != config['formatting']['rag']['empty_message']['vi']
        has_sql_data = sql_summary != config['formatting']['sql']['empty_message']['vi']
        has_dashboard_data = dashboard_summary != config['formatting']['dashboard']['empty_message']['vi']
        has_data = has_rag_data or has_sql_data or has_dashboard_data

        chat_input = (
            f"Query: {query}\n"
            f"Tickers: {json.dumps(tickers or [])}\n"
            f"RAG Summary:\n{rag_summary}\n"
            f"SQL Summary:\n{sql_summary}\n"
            f"Dashboard Summary:\n{dashboard_summary}"
        )
        logger.info(f"Chat input for Chat Completion Agent: {chat_input}")

        try:
            response = chat_completion_agent.run(chat_input)
            token_metrics = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            if isinstance(response, RunResponse):
                metrics = getattr(response, 'metrics', {})
                token_metrics["input_tokens"] = metrics.get('input_tokens', 0)
                token_metrics["output_tokens"] = metrics.get('output_tokens', 0)
                token_metrics["total_tokens"] = metrics.get('total_tokens', token_metrics["input_tokens"] + token_metrics["output_tokens"])
                logger.info(f"[ChatCompletion] Token metrics: Input tokens={token_metrics['input_tokens']}, Output tokens={token_metrics['output_tokens']}, Total tokens={token_metrics['total_tokens']}")
                response = response.content
            # answer_match = re.search(r'Answer: (.*?)\nSummary:', response, re.DOTALL)  # Bỏ phần lấy answer
            summary_match = re.search(r'Summary: (.*)', response, re.DOTALL)
            if not summary_match:
                logger.error(f"Invalid response format from Groq: {response}")
                summary = f"Không có dữ liệu để trả lời truy vấn '{query}'."
            else:
                summary = summary_match.group(1).strip()
        except Exception as e:
            logger.error(f"Error calling Chat Completion Agent: {str(e)}")
            match = re.search(r'\[(.*)\]', sql_response, re.DOTALL)
            if match:
                try:
                    data = json.loads(f"[{match.group(1)}]")
                    if data and 'avg_close_price' in data[0]:
                        summary = f"Dữ liệu từ cơ sở dữ liệu cho thấy giá đóng cửa trung bình của {tickers[0] if tickers else 'Company'} là {data[0]['avg_close_price']} USD. Không có tài liệu RAG để phân tích thêm. Không có dữ liệu biểu đồ cho truy vấn này."
                    elif data and 'sector' in data[0] and 'count' in data[0]:
                        summary = f"Dữ liệu từ cơ sở dữ liệu cho thấy {data[0]['sector']} có {data[0]['count']} công ty, theo sau là {data[1]['sector']} với {data[1]['count']} công ty. Biểu đồ tròn thể hiện rõ phân phối này. Không có tài liệu RAG để phân tích thêm."
                    elif data and 'date' in data[0] and 'close_price' in data[0]:
                        df = pd.DataFrame(data)
                        df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
                        monthly_avg = df.groupby('month')['close_price'].mean().round(2)
                        summary = f"Dữ liệu từ cơ sở dữ liệu cung cấp giá đóng cửa hàng tháng của {tickers[0] if tickers else 'Company'} cho năm 2024. Giá trung bình theo tháng dao động từ {monthly_avg.min()} USD đến {monthly_avg.max()} USD. Biểu đồ boxplot giúp trực quan hóa sự biến động giá theo từng tháng."
                    else:
                        summary = f"Không có dữ liệu để trả lời truy vấn '{query}'."
                except json.JSONDecodeError:
                    summary = f"Không có dữ liệu để trả lời truy vấn '{query}'."
            else:
                summary = f"Không có dữ liệu để trả lời truy vấn '{query}'."

        # Chỉ sử dụng summary trong template
        template = config['output_template']['vi']
        final_response = template.format(
            summary=summary  # Chỉ sử dụng summary
        )
        logger.info(f"Final response: {final_response}")
        return {
            "content": final_response,
            "token_metrics": token_metrics
        }

    except Exception as e:
        logger.error(f"Error in chat completion flow: {str(e)}")
        return {
            "content": f"Không có dữ liệu để trả lời truy vấn '{query}'.",
            "token_metrics": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }