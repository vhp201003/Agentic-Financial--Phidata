# agents/visualize_agent.py
import os
import sys
from pathlib import Path
import json
import yaml
from typing import Dict, Any
import pandas as pd
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from phi.agent import Agent
from phi.model.groq import Groq
from config.env import GROQ_API_KEY, GROQ_MODEL
from utils.logging import setup_logging

logger = setup_logging()

def load_visualization_metadata() -> dict:
    """Load visualization metadata from visualization_metadata.yml."""
    vis_metadata_file = BASE_DIR / "config" / "visualization_metadata.yml"
    try:
        with open(vis_metadata_file, "r") as file:
            vis_metadata = yaml.safe_load(file)
        logger.info("Successfully loaded visualization_metadata.yml")
        return vis_metadata.get("visualization_metadata", [])
    except FileNotFoundError:
        logger.error("visualization_metadata.yml not found")
        return []
    except Exception as e:
        logger.error(f"Error loading visualization metadata: {str(e)}")
        return []

def create_visualize_agent() -> Agent:
    """Create Visualize Agent to analyze SQL data and generate chart configurations."""
    logger.info("Creating Visualize Agent")
    visualization_metadata = load_visualization_metadata()
    vis_metadata_json = json.dumps(visualization_metadata, ensure_ascii=False, indent=2)

    system_prompt = f"""
You are Visualize Agent, responsible for analyzing SQL data and generating JSON configurations for data visualization. Your task is to determine the appropriate chart type and column mappings based on the user query and input data. Prioritize the chart type specified in the query (e.g., 'pie chart', 'bar chart', 'scatter plot') and ensure compatibility with the data. Use a balanced, general approach to infer chart type when the query is ambiguous, considering both query keywords and data structure. Return ONLY a JSON object.

1. Analyze User Query:
   - Extract chart type from query (e.g., 'pie chart', 'bar chart', 'line chart', 'scatter plot', 'histogram', 'boxplot', 'heatmap'). Assign high priority to the specified chart type.
   - Identify key terms (e.g., 'distribution', 'monthly', 'market capitalization', 'p/e ratio', 'top companies') to guide chart type and column mapping.
   - Example: Query 'Create a scatter plot of market capitalization versus P/E ratio' indicates 'scatter' with 'market_cap' and 'pe_ratio'.

2. Analyze Input Data:
   - Input: SQL data (JSON list of dictionaries) and query.
   - Determine column types:
     - Month: Columns named 'month' or with numeric values (e.g., 1.0, 2.0) or string values matching YYYY-MM.
     - Date: Columns named 'date' or with values matching YYYY-MM-DD.
     - Numeric: Columns with float or integer values (e.g., 'market_cap', 'pe_ratio', 'close_price', 'count').
     - Categorical: Columns with string values (e.g., 'sector', 'symbol', 'industry').
   - Identify available columns and their roles based on visualization_metadata:
     {vis_metadata_json}

3. Select Chart Type:
   - If query specifies a chart type (e.g., 'bar chart'), prioritize it if data is compatible:
     - 'pie_chart': Requires Categorical + Numeric (e.g., 'sector' and 'count').
     - 'bar_chart': Requires Categorical/Month + Numeric (e.g., 'symbol' and 'market_cap').
     - 'scatter': Requires Numeric + Numeric (e.g., 'market_cap' and 'pe_ratio').
     - 'line_chart': Requires Date + Numeric (e.g., 'date' and 'close_price').
     - 'histogram': Requires Single Numeric (e.g., 'daily_return').
     - 'boxplot': Requires Date/Month + Numeric with grouping (e.g., 'month' and 'close_price').
     - 'heatmap': Requires multiple Numeric columns (e.g., 'symbol1_return', 'symbol2_return').
   - If query is ambiguous or data is incompatible, infer based on data and query keywords:
     - Categorical + Numeric: Prefer 'bar_chart' unless 'distribution' or 'proportion' in query, then 'pie_chart'.
     - Month + Numeric: Prefer 'bar_chart'.
     - Date + Numeric: Prefer 'line_chart'.
     - Numeric + Numeric: Prefer 'scatter'.
     - Single Numeric: Prefer 'histogram'.
     - Date/Month + Numeric with grouping: Prefer 'boxplot'.
     - Multiple Numeric: Prefer 'heatmap'.
   - Balance query keywords and data compatibility. Do not favor any chart type unless supported by both query and data.

4. Map Columns to Chart Requirements:
   - Use the 'columns' section in visualization_metadata to map data columns to roles based on intent_keywords.
   - Examples:
     - For pie_chart: Map 'sector' to category_col, 'count' or 'proportion' to value_col.
     - For bar_chart: Map 'symbol' or 'month' to category_col, 'market_cap' or 'avg_close_price' to value_col.
     - For scatter: Map 'market_cap' to x_col, 'pe_ratio' to y_col, 'symbol' to label_col if available.
     - For line_chart: Map 'date' to x_col, 'close_price' to y_col, check for 'additional_lines' (e.g., 'rolling_avg').

5. Validate Configuration:
   - Ensure all required columns for the selected chart type are present in the data.
   - If missing, return an error JSON with a message.

6. Output:
   - JSON object with:
     - type: Chart type (e.g., 'scatter').
     - Configuration fields based on chart type (e.g., x_col, y_col for scatter).
     - error: null if successful, or an error message if validation fails.
   - Example for scatter:
     {{"type": "scatter", "x_col": "market_cap", "y_col": "pe_ratio", "label_col": "symbol", "error": null}}
   - Example for bar_chart:
     {{"type": "bar_chart", "category_col": "symbol", "value_col": "market_cap", "error": null}}
   - Example for pie_chart:
     {{"type": "pie_chart", "category_col": "sector", "value_col": "count", "error": null}}

7. Error Handling:
   - If data is empty or invalid, return: {{"type": null, "error": "No valid data provided"}}.
   - If chart type cannot be determined, return: {{"type": null, "error": "Cannot determine chart type"}}.
   - If columns do not match the chart type, return: {{"type": null, "error": "Data incompatible with requested chart type"}}.

Input Examples:
- Query: "Create a pie chart showing the distribution of DJIA companies by sector"
  Data: [{{"sector": "Technology", "count": 6}}, {{"sector": "Healthcare", "count": 5}}, ...]
  Output: {{"type": "pie_chart", "category_col": "sector", "value_col": "count", "error": null}}
- Query: "Create a scatter plot of market capitalization versus P/E ratio for all DJIA companies"
  Data: [{{"market_cap": 3143825096704, "pe_ratio": 33.22}}, {{"market_cap": 150868656128, "pe_ratio": 37.2}}, ...]
  Output: {{"type": "scatter", "x_col": "market_cap", "y_col": "pe_ratio", "label_col": "symbol", "error": null}}
- Query: "Create a bar chart of the top 10 companies by market capitalization"
  Data: [{{"symbol": "AAPL", "market_cap": 3143825096704}}, {{"symbol": "MSFT", "market_cap": 2913004945408}}, ...]
  Output: {{"type": "bar_chart", "category_col": "symbol", "value_col": "market_cap", "error": null}}
"""
    return Agent(
        model=Groq(
            id="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            timeout=30,
            max_retries=5,
            temperature=0.3,
            max_tokens=500,
            top_p=0.9,
            response_format={"type": "json_object"}
        ),
        system_prompt=system_prompt,
        custom_run=lambda self, input_data: self.run_with_validation(input_data),
        # debug_mode=True,
    )

def run_with_validation(self, input_data: Any) -> Dict[str, Any]:
    """Validate input and run the Visualize Agent."""
    try:
        # Handle input as either string (JSON) or dictionary
        if isinstance(input_data, str):
            try:
                input_dict = json.loads(input_data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse input JSON: {str(e)}")
                return {"type": None, "error": "Invalid JSON input"}
        elif isinstance(input_data, dict):
            input_dict = input_data
        else:
            logger.error(f"Invalid input type: {type(input_data)}")
            return {"type": None, "error": "Invalid input format, expected JSON string or dictionary"}

        data = input_dict.get("data", [])
        query = input_dict.get("query", "")

        if not isinstance(data, list) or not data:
            logger.error("Invalid or empty data provided")
            return {"type": None, "error": "No valid data provided"}

        # Convert data to DataFrame for analysis
        df = pd.DataFrame(data)
        if df.empty:
            logger.error("DataFrame is empty")
            return {"type": None, "error": "No valid data provided"}

        # Prepare input for the agent as a JSON string
        agent_input = json.dumps({
            "data": data,
            "query": query,
            "columns": list(df.columns)
        }, ensure_ascii=False)

        # Run the agent
        response = self.run(agent_input)
        if isinstance(response, dict):
            return response
        elif isinstance(response, str):
            return json.loads(response)
        else:
            logger.error(f"Unexpected response type: {type(response)}")
            return {"type": None, "error": "Invalid response from agent"}

    except Exception as e:
        logger.error(f"Error in Visualize Agent: {str(e)}")
        return {"type": None, "error": f"Error processing visualization: {str(e)}"}