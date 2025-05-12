import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import unittest
from agents.finance_agent import create_finance_agent

class TestFinanceAgent(unittest.TestCase):
    def test_finance_agent_creation(self):
        agent = create_finance_agent()
        self.assertEqual(agent.name, "Finance Agent")