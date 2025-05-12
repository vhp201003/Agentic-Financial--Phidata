import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import unittest
from agents.text_to_sql_agent import create_text_to_sql_agent
from tools.sql_tool import CustomSQLTool

class TestSQLAgent(unittest.TestCase):
    def test_sql_agent_creation(self):
        sql_tool = CustomSQLTool()
        agent = create_text_to_sql_agent(sql_tool)
        self.assertEqual(agent.name, "TextToSQL Agent")