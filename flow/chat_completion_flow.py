# flow/chat_completion_flow.py
import json
import yaml
from pathlib import Path
from phi.agent import Agent, RunResponse
from utils.logging import setup_logging
import re

BASE_DIR = Path(__file__).resolve().parent.parent
logger = setup_logging()

def load_config() -> dict:
    config_file = Path.joinpath(BASE_DIR, "config/chat_completion_config.yml")
    logger.info(f"Loading chat completion config from {config_file}")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    logger.info("Successfully loaded chat completion config")
    return config

def prepare_rag_summary(rag_documents: list, config: dict) -> str:
    if not rag_documents or not all(isinstance(doc, dict) and 'document' in doc and 'filename' in doc and 'company' in doc for doc in rag_documents):
        return config['formatting']['rag']['empty_message']['vi']

    rag_by_company = {}
    financial_metrics_pattern = re.compile(r'(Net revenue|Net income|Operating expenses|Diluted.*earnings per share|Total volume|Payments volume|Transactions processed)\s*(FY\s*\d{4})?\s*[:=]?\s*\$?([\d,.]+[TBM]?|\d+\.\d+[TBM]?)', re.IGNORECASE)

    for doc in rag_documents:
        company = doc['company']
        if company not in rag_by_company:
            rag_by_company[company] = []
        
        content = doc['document']
        metrics = financial_metrics_pattern.findall(content)
        
        metrics_by_year = {}
        for metric, year, value in metrics:
            year = year.strip() if year else "Unknown Year"
            if year not in metrics_by_year:
                metrics_by_year[year] = []
            value = value.replace(',', '').replace('$', '')
            metrics_by_year[year].append(f"{metric}: {value}")
        
        formatted_metrics = []
        for year, metric_list in metrics_by_year.items():
            formatted_metrics.append(f"{year}: {', '.join(metric_list)}")
        
        if not formatted_metrics:
            content = content[:1000] + ("..." if len(content) > 1000 else "")
            formatted_metrics = [content]
        
        rag_by_company[company].append(f"{company}: {'; '.join(formatted_metrics)} from {doc['filename']}")

    return "\n".join(f"{company}: " + "; ".join(entries) for company, entries in rag_by_company.items())

def prepare_sql_summary(sql_response: str, config: dict, tickers: list) -> str:
    # Nếu sql_response không chứa dữ liệu, trả về empty message
    if "Dữ liệu từ cơ sở dữ liệu" not in sql_response:
        return config['formatting']['sql']['empty_message']['vi']

    # Trích xuất JSON data từ sql_response
    match = re.search(r'\[(.*)\]', sql_response, re.DOTALL)
    if not match:
        logger.warning(f"No JSON data found in sql_response: {sql_response}")
        return config['formatting']['sql']['empty_message']['vi']

    try:
        json_str = match.group(1)
        data = json.loads(f"[{json_str}]")
        if not data:
            return config['formatting']['sql']['empty_message']['vi']

        summaries = []
        for record in data:
            # Kiểm tra record có dữ liệu hợp lệ
            if not isinstance(record, dict):
                logger.warning(f"Invalid record format in SQL data: {record}")
                continue
            for key, value in record.items():
                # Format key và value, đảm bảo không bỏ qua single-value như 'min'
                formatted_key = key.replace('_', ' ').title()
                formatted_value = str(value) if value is not None else "N/A"
                summaries.append(f"{formatted_key}: {formatted_value}")
        if not summaries:
            logger.warning(f"No valid summaries generated from SQL data: {data}")
            return config['formatting']['sql']['empty_message']['vi']
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

def chat_completion_flow(query: str, rag_documents: list, sql_response: str, dashboard_info: dict, chat_completion_agent: Agent, tickers: list = None) -> dict:
    config = load_config()
    try:
        # Prepare input for chat completion
        rag_summary = prepare_rag_summary(rag_documents, config)
        sql_summary = sql_response #prepare_sql_summary(sql_response, config, tickers)
        dashboard_summary = prepare_dashboard_summary(dashboard_info, config)

        # Kiểm tra nếu không có dữ liệu từ bất kỳ nguồn nào
        has_data = (
            rag_summary != config['formatting']['rag']['empty_message']['vi'] or
            sql_summary != config['formatting']['sql']['empty_message']['vi'] or
            dashboard_summary != config['formatting']['dashboard']['empty_message']['vi']
        )
        if not has_data:
            logger.error("No valid data from RAG, SQL, or Dashboard")
            return {
                "content": f"# Phản hồi\n## Tóm tắt\nKhông có dữ liệu để trả lời truy vấn '{query}'.",
                "token_metrics": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            }

        input_data = (
            f"Query: {query}\n"
            f"Tickers: {json.dumps(tickers or [])}\n"
            f"RAG Summary: {rag_summary[:1000]}...\n"
            f"SQL Summary: {sql_summary[:1000]}...\n"
            f"Dashboard Summary: {dashboard_summary[:1000]}..."
        )
        logger.info(f"Chat input for Chat Completion Agent: {input_data[:500]}...")

        # Run chat completion agent
        response = chat_completion_agent.run(input_data)
        logger.debug(f"Raw response from Groq: {response}")

        # Handle response
        if isinstance(response, RunResponse):
            response_content = response.content if response.content else config['formatting']['chat']['empty_message']['vi']
        elif isinstance(response, dict) and 'content' in response:
            response_content = response['content']
        elif isinstance(response, str):
            response_content = response  # Giữ nguyên Markdown, không strip
        else:
            logger.error(f"Unexpected response type from Groq: {type(response)}")
            response_content = config['formatting']['chat']['empty_message']['vi']

        # Extract metrics if available
        token_metrics = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if isinstance(response, RunResponse) and hasattr(response, 'metrics'):
            metrics = response.metrics
            token_metrics["input_tokens"] = metrics.get('input_tokens', 0)
            token_metrics["output_tokens"] = metrics.get('output_tokens', 0)
            token_metrics["total_tokens"] = metrics.get('total_tokens', token_metrics["input_tokens"] + token_metrics["output_tokens"])
        elif isinstance(response, dict) and 'metrics' in response:
            metrics = response.get('metrics', {})
            token_metrics["input_tokens"] = metrics.get('input_tokens', 0)
            token_metrics["output_tokens"] = metrics.get('output_tokens', 0)
            token_metrics["total_tokens"] = metrics.get('total_tokens', token_metrics["input_tokens"] + token_metrics["output_tokens"])

        logger.info(f"[ChatCompletion] Token metrics: Input tokens={token_metrics['input_tokens']}, Output tokens={token_metrics['output_tokens']}, Total tokens={token_metrics['total_tokens']}")

        return {
            "content": response_content,
            "token_metrics": token_metrics
        }

    except Exception as e:
        logger.error(f"Error in chat_completion_flow: {str(e)}")
        return {
            "content": f"# Phản hồi\n## Tóm tắt\nKhông có dữ liệu để trả lời truy vấn '{query}'.",
            "token_metrics": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }