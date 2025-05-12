import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import unittest
from agents.rag_agent import create_rag_agent
from tools.rag_tool import CustomRAGTool

class TestRAGAgent(unittest.TestCase):
    def test_rag_agent_creation(self):
        rag_tool = CustomRAGTool()
        agent = create_rag_agent(rag_tool)
        self.assertEqual(agent.name, "RAG Agent")