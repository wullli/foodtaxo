import inspect
import unittest
from os import path
from pathlib import Path

from dspy.teleprompt.copro_optimizer import BasicGenerateInstruction
from llm_food_taxonomy.util import dspy_to_chat

resource_dir = Path(path.join(path.dirname(path.abspath(__file__)), 'resources'))


class TestChatFormat(unittest.TestCase):

    def setUp(self):
        with open(resource_dir / "zero_example_parents.txt") as f:
            self.zero_parents_prompt = f.read()

        with open(resource_dir / "zero_example_parents_retry.txt") as f:
            self.zero_retry_parents_prompt = f.read()

        with open(resource_dir / "zero_example_children.txt") as f:
            self.zero_children_prompt = f.read()

        with open(resource_dir / "few_example_parents.txt") as f:
            self.few_parents_prompt = f.read()

    def _check_chat_format(self, messages, expected_roles, expected_start):
        self.assertTrue(len(messages) == len(expected_roles) == len(expected_start))
        exp = zip(expected_roles, expected_start)

        for m, (exp_role, exp_start) in zip(messages, exp):
            role = m["role"]
            content_start = m["content"].strip()[:len(exp_start)]
            self.assertEqual(role, exp_role)
            self.assertEqual(content_start, exp_start)

    def test_zero_parents_chat_format(self):
        messages = dspy_to_chat(self.zero_parents_prompt)
        exp_roles = ["system", "user", "assistant"]
        exp_start_words = ["Which are the most specific parent concepts",
                           "Context:", "Reasoning: Let's think step by step in order to"]

        self._check_chat_format(messages, exp_roles, exp_start_words)

    def test_zero_children_chat_format(self):
        messages = dspy_to_chat(self.zero_children_prompt)
        exp_roles = ["system", "user", "assistant"]
        exp_start_words = ["Which of the candidates are child concepts",
                           "Context:", "Reasoning: Let's think step by step in order to"]

        self._check_chat_format(messages, exp_roles, exp_start_words)

    def test_zero_retry_chat_format(self):
        messages = dspy_to_chat(self.zero_retry_parents_prompt)
        exp_roles = ["system", "user", "assistant"]
        exp_start_words = ["Which are the most specific parent concepts",
                           "Context:", "Reasoning: Let's think step by step in order to"]

        self._check_chat_format(messages, exp_roles, exp_start_words)

    def test_few_parents_chat_format(self):
        messages = dspy_to_chat(self.few_parents_prompt)
        exp_roles = ["system", "user", "assistant", "user", "assistant", "user", "assistant"]
        exp_start_words = ["Which are the most specific parent concepts",
                           "Context:",
                           "Reasoning: Let's think step by step in order to",
                           "Context:",
                           "Reasoning: Let's think step by step in order to",
                           "Context:",
                           "Reasoning: Let's think step by step in order to"]
        self._check_chat_format(messages, exp_roles, exp_start_words)

    def test_optimizer_prompts(self):
        prompt = inspect.cleandoc(BasicGenerateInstruction.__doc__)
        messages = dspy_to_chat(prompt)
        exp_roles = ["user"]
        exp_start_words = ["You are an instruction optimizer for large language models."]
        self._check_chat_format(messages, exp_roles, exp_start_words)
