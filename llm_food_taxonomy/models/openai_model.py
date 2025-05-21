import json
from typing import Any

from dsp import GPT3, chat_request, completions_request

from llm_food_taxonomy.util import dspy_to_chat


class ConversationGPT3(GPT3):

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            # caching mechanism requires hashable kwargs
            try:
                messages = dspy_to_chat(prompt)
            except Exception as e:
                messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            kwargs["messages"] = messages
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = chat_request(**kwargs)

        else:
            kwargs["prompt"] = prompt
            response = completions_request(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response
