# flow/orchestrator_flow.py
import json
from pathlib import Path
from phi.agent import Agent, RunResponse
from utils.logging import setup_logging, get_collected_logs
from utils.response import standardize_response
from utils.response_parser import parse_response_to_json
from flow.sql_flow import sql_flow
from flow.rag_flow import rag_flow
from flow.chat_completion_flow import chat_completion_flow
import re
import yaml
from agents.visualize_agent import create_visualize_agent
from agents.rag_agent import run_rag_agent

BASE_DIR = Path(__file__).resolve().parent.parent
logger = setup_logging()

def load_metadata() -> dict:
    """Load visualized templates."""
    metadata = {"template_query": []}
    vis_template_file = BASE_DIR / "config" / "visualized_template.yml"
    try:
        with open(vis_template_file, "r") as file:
            vis_template = yaml.safe_load(file)
        logger.info("Successfully loaded visualized_template.yml")
        metadata["visualized_template"] = vis_template["visualized_template"]
    except FileNotFoundError:
        logger.error("visualized_template.yml not found")
        metadata["visualized_template"] = []
    except Exception as e:
        logger.error(f"Error loading visualized_template: {str(e)}")
        metadata["visualized_template"] = []
    return metadata

def process_response(response: any, context: str) -> tuple[dict, dict]:
    """Process response, extract token metrics, and return JSON dict with metrics."""
    token_metrics = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        if isinstance(response, RunResponse):
            metrics = getattr(response, 'metrics', {})
            input_tokens = metrics.get('input_tokens', 0)
            output_tokens = metrics.get('output_tokens', 0)
            token_metrics["input_tokens"] = input_tokens[0] if isinstance(input_tokens, list) and input_tokens else input_tokens
            token_metrics["output_tokens"] = output_tokens[0] if isinstance(output_tokens, list) and output_tokens else output_tokens
            token_metrics["total_tokens"] = metrics.get('total_tokens', token_metrics["input_tokens"] + token_metrics["output_tokens"])
            if isinstance(token_metrics["total_tokens"], list):
                token_metrics["total_tokens"] = token_metrics["total_tokens"][0] if token_metrics["total_tokens"] else 0
            logger.info(f"[{context}] Token metrics: Input tokens={token_metrics['input_tokens']}, Output tokens={token_metrics['output_tokens']}, Total tokens={token_metrics['total_tokens']}")
            response_content = response.content
        else:
            logger.error(f"Unexpected response type in {context}: {type(response)}, content: {response}")
            response_content = response
            
        if isinstance(response_content, dict):
            logger.info(f"[{context}] Response is already JSON: {json.dumps(response_content, ensure_ascii=False)}")
            return response_content, token_metrics
        elif not isinstance(response_content, str):
            logger.warning(f"[{context}] Unexpected result type: {type(response_content)}")
            return standardize_response("error", "Sorry, the system cannot parse your query. Please try again with a different query.", {}), token_metrics

        cleaned_content = re.sub(r'```(?:json|sql)?|```|\n|\t', '', response_content).strip()
        logger.debug(f"[{context}] Cleaned response content: {cleaned_content}")
        return parse_response_to_json(cleaned_content, context), token_metrics

    except Exception as e:
        logger.error(f"[{context}] Error processing response: {str(e)}")
        return standardize_response("error", "Sorry, the system cannot parse your query. Please try again with a different query.", {}), token_metrics

def limit_sql_records(sql_response: str, max_records: int = 5) -> str:
    """Limit the number of records in SQL response."""
    try:
        match = re.search(r'\[(.*)\]', sql_response, re.DOTALL)
        if not match:
            return sql_response

        records_str = match.group(1)
        try:
            records = json.loads(f"[{records_str}]")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in SQL response: {str(e)}")
            return sql_response

        if len(records) <= max_records:
            return sql_response

        limited_records = records[:max_records]
        limited_records_str = json.dumps(limited_records, ensure_ascii=False)
        return sql_response[:match.start()] + limited_records_str + sql_response[match.end():]
    except Exception as e:
        logger.error(f"Error limiting SQL records: {str(e)}")
        return sql_response

def limit_records(data: list, max_records: int = 5, for_dashboard: bool = False, for_chat_input: bool = False) -> list:
    """Limit the number of records in the data list for specific purposes."""
    if not isinstance(data, list):
        logger.error(f"Expected list for limiting records, got {type(data)}")
        return []
    
    if for_dashboard:
        max_dashboard_records = 10  # Reduced for Visualize Agent
        if data and 'symbol' in data[0]:
            unique_symbols = list(dict.fromkeys([record['symbol'] for record in data]))
            filtered_data = [record for record in data if record['symbol'] in unique_symbols]
            if len(filtered_data) > max_dashboard_records:
                logger.info(f"Limiting {len(filtered_data)} dashboard records to {max_dashboard_records} for Visualize Agent")
                return filtered_data[:max_dashboard_records]
            return filtered_data
        if len(data) > max_dashboard_records:
            logger.info(f"Limiting {len(data)} dashboard records to {max_dashboard_records} for Visualize Agent")
            return data[:max_dashboard_records]
        return data
    
    if for_chat_input:
        if len(data) > max_records:
            logger.info(f"Limiting {len(data)} records to {max_records} for Chat Completion Agent")
            return data[:max_records]
        return data
    
    return data  # No limit for other cases

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
    # Regex để tìm các chỉ số tài chính kèm năm (ví dụ: "Net revenue FY 2022: $29,310")
    financial_metrics_pattern = re.compile(r'(Net revenue|Net income|Operating expenses|Diluted.*earnings per share|Total volume|Payments volume|Transactions processed)\s*(FY\s*\d{4})?\s*[:=]?\s*\$?([\d,.]+[TBM]?|\d+\.\d+[TBM]?)', re.IGNORECASE)

    for doc in rag_documents:
        company = doc['company']
        if company not in rag_by_company:
            rag_by_company[company] = []
        
        # Trích xuất các chỉ số tài chính
        content = doc['document']
        metrics = financial_metrics_pattern.findall(content)
        
        # Nhóm dữ liệu theo năm
        metrics_by_year = {}
        for metric, year, value in metrics:
            year = year.strip() if year else "Unknown Year"
            if year not in metrics_by_year:
                metrics_by_year[year] = []
            # Chuẩn hóa giá trị: loại bỏ dấu phẩy và ký tự không cần thiết
            value = value.replace(',', '').replace('$', '')
            metrics_by_year[year].append(f"{metric}: {value}")
        
        # Định dạng lại metrics theo năm
        formatted_metrics = []
        for year, metric_list in metrics_by_year.items():
            formatted_metrics.append(f"{year}: {', '.join(metric_list)}")
        
        # Nếu không tìm thấy chỉ số, lấy tối đa 1000 ký tự
        if not formatted_metrics:
            content = content[:1000] + ("..." if len(content) > 1000 else "")
            formatted_metrics = [content]
        
        rag_by_company[company].append(f"{company}: {'; '.join(formatted_metrics)} from {doc['filename']}")

    return "\n".join(f"{company}: " + "; ".join(entries) for company, entries in rag_by_company.items())

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

def orchestrator_flow(query: str, orchestrator: Agent, sql_agent, sql_tool, rag_tool, chat_completion_agent, thinking_queue=None, chat_history=None) -> dict:
    metadata = load_metadata()
    visualize_agent = create_visualize_agent()
    
    # Initialize token metrics dictionary
    token_metrics = {
        "orchestrator": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "text2sql": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "visualize": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "chat_completion": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    }
    
    try:
        # Push thinking message
        if thinking_queue:
            thinking_queue.put("Đang phân tích query...")

        # Process Orchestrator response
        input_data = {"query": query, "chat_history": chat_history or []}
        result, token_metrics["orchestrator"] = process_response(orchestrator.run(json.dumps(input_data)), "Orchestrator")
        result_dict = result
        logger.info(f"Orchestrator Response: {json.dumps(result_dict, indent=2, ensure_ascii=False)}")
        if thinking_queue:
            thinking_queue.put(f"Orchestrator: Phân tích truy vấn: {json.dumps(result_dict, ensure_ascii=False)[:200]}...")

        if result_dict.get("status") == "error":
            return {
                "status": "error",
                "message": "Sorry, the system cannot parse your query. Please try again with a different query, e.g., 'Stock price of Apple on 01/01/2025'.",
                "data": {
                    "token_metrics": token_metrics
                },
                "logs": get_collected_logs()
            }

        if thinking_queue:
            thinking_queue.put("Đang xác định các agent cần dùng...")

        data = result_dict.get("data", {})
        tickers = data.get("tickers", [])  # Định nghĩa tickers từ data
        rag_documents = []
        sql_response = "No response from SQL."
        actual_results = []

        for agent_name in data.get("agents", []):
            sub_query = data.get("sub_queries", {}).get(agent_name)
            if not sub_query:
                logger.error(f"No sub-query provided for {agent_name}")
                return {
                    "status": "error",
                    "message": "Sorry, the system cannot process your request. Please try again with a different request.",
                    "data": {
                        "token_metrics": token_metrics
                    },
                    "logs": get_collected_logs()
                }
            if agent_name == "text2sql_agent":
                if thinking_queue:
                    thinking_queue.put("Đang sinh SQL query...")
                metadata_with_columns = {
                    "tickers": tickers,
                    "date_range": data.get("date_range"),
                    "visualized_template": metadata["visualized_template"]
                }
                final_response = sql_flow(sub_query, sql_agent, sql_tool, metadata=metadata_with_columns)
                response_for_chat = final_response["response_for_chat"]
                actual_result = final_response["actual_result"]
                sql_response = limit_sql_records(response_for_chat, max_records=5)
                dashboard_result = actual_result
                limited_result = limit_records(actual_result, max_records=5, for_dashboard=False)
                if thinking_queue:
                    sql_query = final_response.get("sql_query", "Không có câu SQL cụ thể.")
                    thinking_queue.put(f"SQL: {sql_query}")
                    thinking_queue.put(f"Kết quả SQL: {json.dumps(actual_result, ensure_ascii=False)[:200]}...")
                logger.info(f"SQL Response (limited for log): {sql_response}")
                logger.info(f"Dashboard records: {len(dashboard_result)}, Limited log records: {len(limited_result)}")
                actual_results.append(dashboard_result)
                token_metrics["text2sql"] = final_response.get("token_metrics", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

            elif agent_name == "rag_agent":
                if thinking_queue:
                    thinking_queue.put("Đang phân tích query cho RAG...")
                # Gọi RAG Agent để tạo sub-query và xác định company
                rag_agent_result = run_rag_agent(sub_query)
                logger.info(f"RAG Agent result: {rag_agent_result}")
                optimized_sub_query = rag_agent_result.get("sub-query", sub_query)
                company = rag_agent_result.get("company", None)
                
                if thinking_queue:
                    thinking_queue.put("Đang tìm kiếm tài liệu RAG...")
                # Truyền optimized_sub_query và company vào rag_flow
                rag_documents = rag_flow(optimized_sub_query, rag_tool, company=company)
                if thinking_queue:
                    thinking_queue.put(f"RAG: {json.dumps(rag_documents, ensure_ascii=False)[:200]}...")
                logger.info(f"RAG Documents: {rag_documents}")

        if thinking_queue:
            thinking_queue.put("Đang chuẩn bị visualization...")

        dashboard_enabled = data.get("Dashboard", False) and bool(actual_results and actual_results[0])
        dashboard_data = actual_results[0] if actual_results else []
        limited_dashboard_data = limit_records(dashboard_data, max_records=5, for_chat_input=True)
        vis_dashboard_data = limit_records(dashboard_data, for_dashboard=True)

        visualization_config = {}
        if dashboard_enabled:
            vis_input = {
                "data": vis_dashboard_data,
                "query": query
            }
            vis_input_str = json.dumps(vis_input, ensure_ascii=False)
            vis_response = visualize_agent.run(vis_input_str)
            if isinstance(vis_response, RunResponse):
                visualization_config = vis_response.content
                if isinstance(visualization_config, str):
                    visualization_config = json.loads(visualization_config)
                metrics = getattr(vis_response, 'metrics', {})
                token_metrics["visualize"]["input_tokens"] = metrics.get('input_tokens', 0)
                token_metrics["visualize"]["output_tokens"] = metrics.get('output_tokens', 0)
                token_metrics["visualize"]["total_tokens"] = metrics.get('total_tokens', token_metrics["visualize"]["input_tokens"] + token_metrics["visualize"]["output_tokens"])
                logger.info(f"[Visualize] Token metrics: Input tokens={token_metrics['visualize']['input_tokens']}, Output tokens={token_metrics['visualize']['output_tokens']}, Total tokens={token_metrics['visualize']['total_tokens']}")
            else:
                visualization_config = vis_response
            if thinking_queue:
                thinking_queue.put(f"Visualized: {json.dumps(visualization_config, ensure_ascii=False)[:200]}...")
            logger.info(f"Visualize Agent config: {json.dumps(visualization_config, ensure_ascii=False)}")
            if visualization_config.get("error"):
                logger.error(f"Visualize Agent error: {visualization_config['error']}")
                dashboard_enabled = False

        dashboard_info = {
            "enabled": dashboard_enabled,
            "data": limited_dashboard_data,
            "visualization": {
                "type": visualization_config.get("type", "none"),
                "required_columns": visualization_config.get("required_columns", []),
                "aggregation": None,
                "ui_requirements": visualization_config if visualization_config.get("type") else {}
            }
        }

        #Sửa đoạn chat_completion_flow
        if thinking_queue:
            thinking_queue.put("Đang sinh câu trả lời cuối...")
        final_response_message = chat_completion_flow(query, rag_documents, sql_response, dashboard_info, chat_completion_agent, tickers=tickers)

        # Xử lý phản hồi từ chat_completion_flow
        if isinstance(final_response_message, dict):
            final_response_message_dict = final_response_message
        else:
            final_response_message_dict = {"content": str(final_response_message), "token_metrics": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}

        # Đảm bảo token_metrics hợp lệ
        token_metrics["chat_completion"] = final_response_message_dict.get("token_metrics", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        if not isinstance(token_metrics["chat_completion"], dict):
            token_metrics["chat_completion"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        if thinking_queue:
            chat_input = (
                f"Query: {query}\n"
                f"Tickers: {json.dumps(tickers)}\n"
                f"RAG Summary: {prepare_rag_summary(rag_documents, load_config())[:200]}...\n"
                f"SQL Summary: {prepare_sql_summary(sql_response, load_config(), tickers)[:200]}...\n"
                f"Dashboard Summary: {prepare_dashboard_summary(dashboard_info, load_config())[:200]}..."
            )
            thinking_queue.put(f"Chat Completion Input: {chat_input}")
            thinking_queue.put(f"Chat Completion Output: {final_response_message_dict.get('content', 'No response')[:200]}...")
        logger.info(f"Final response message: {final_response_message_dict.get('content', final_response_message)}")

        final_dashboard_info = {
            "enabled": dashboard_enabled,
            "data": dashboard_data,
            "visualization": {
                "type": visualization_config.get("type", "none"),
                "required_columns": visualization_config.get("required_columns", []),
                "aggregation": None,
                "ui_requirements": visualization_config if visualization_config.get("type") else {}
            }
        }
        final_response = {
            "status": "success",
            "message": final_response_message_dict.get("content", "Không có phản hồi chi tiết."),
            "data": {
                "result": final_response_message_dict.get("content", "Không có phản hồi chi tiết."),
                "dashboard": final_dashboard_info,
                "token_metrics": token_metrics
            },
            "logs": get_collected_logs()
        }
        return final_response

    except Exception as e:
        logger.error(f"Error in orchestrator flow: {str(e)}")
        return {
            "status": "error",
            "message": "Sorry, the system cannot process your request. Please try again with a different request.",
            "data": {
                "token_metrics": token_metrics
            },
            "logs": get_collected_logs()
        }