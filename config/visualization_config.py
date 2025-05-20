# config/visualization_config.py
VISUALIZATION_CONFIG = {
    "table": {
        "required_columns": [],  # Bất kỳ cột nào trong dữ liệu
        "default_columns": ["symbol", "date", "close_price"]  # Cột mặc định nếu rỗng
    },
    "time_series": {
        "required_columns": ["date", "close_price"],
        "default_columns": ["date", "close_price"]
    },
    "line_chart": {
        "required_columns": ["date", "close_price"],
        "default_columns": ["date", "close_price"]
    },
    "bar_chart": {
        "required_columns": ["symbol", "close_price"],
        "default_columns": ["symbol", "close_price"]
    },
    "pie_chart": {
        "required_columns": ["sector", "count"],  # Default to 'count' for pie_chart_count
        "default_columns": ["sector", "count", "proportion"]  # Support both count and proportion
    },
    "histogram": {
        "required_columns": ["close_price"],
        "default_columns": ["close_price"]
    },
    "boxplot": {
        "required_columns": ["month", "close_price"],
        "default_columns": ["month", "close_price"]
    },
    "scatter": {
        "required_columns": ["market_cap", "pe_ratio"],
        "default_columns": ["market_cap", "pe_ratio"]
    },
    "heatmap": {
        "required_columns": [],  # Tùy thuộc vào dữ liệu
        "default_columns": ["aapl_return", "msft_return"]  # Ví dụ cho heatmap
    }
}