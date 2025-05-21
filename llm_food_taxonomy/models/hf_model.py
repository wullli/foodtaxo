import logging
import os
import pdb
import re
from typing import Optional, Literal

from dsp.modules.hf import openai_to_hf
from dspy import HFModel
from transformers import BitsAndBytesConfig

from llm_food_taxonomy.util import dspy_to_chat


class FastHFModel(HFModel):

    def __init__(self, model: str,
                 checkpoint: Optional[str] = None,
                 is_client: bool = False,
                 hf_device_map: Literal["auto", "balanced", "balanced_low_0", "sequential"] = "auto",
                 token=None,
                 quantize=0,
                 max_len=None,
                 **kwargs):
        """wrapper for Hugging Face models

        Args:
            model (str): HF model identifier to load and use
            checkpoint (str, optional): load specific checkpoints of the model. Defaults to None.
            is_client (bool, optional): whether to access models via client. Defaults to False.
            hf_device_map (str, optional): HF config strategy to load the model.
                Recommeded to use "auto", which will help loading large models using accelerate. Defaults to "auto".
        """

        super().__init__(model, checkpoint=checkpoint, hf_device_map=hf_device_map, is_client=True)
        for k, v in kwargs.items():
            self.kwargs[k] = v
        self.provider = "hf"
        self.is_client = is_client
        self.device_map = hf_device_map
        self.model_name = model
        self.llama3_re = re.compile(r"<\|start_header_id\|>.+<\|end_header_id\|>|<\|.+\|>")
        if token is None:
            token = os.getenv("HF_TOKEN")
        if not self.is_client:
            try:
                import torch
                from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "You need to install Hugging Face transformers library to use HF models.",
                ) from exc
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            qconfig = BitsAndBytesConfig(load_in_8bit=True if quantize == 8 else False,
                                         load_in_4bit=True if quantize == 4 else False) if quantize != 0 else None

            hf_kwargs = dict(trust_remote_code=True,
                             token=token,
                             torch_dtype=torch.float16 if "gemma" not in self.model_name else torch.float32,
                             quantization_config=qconfig)
            tok_kwargs = dict()
            if max_len is not None:
                tok_kwargs["model_max_length"] = max_len

            try:
                architecture = AutoConfig.from_pretrained(model).__dict__["architectures"][0]
                self.encoder_decoder_model = ("ConditionalGeneration" in architecture) or (
                        "T5WithLMHeadModel" in architecture)
                self.decoder_only_model = ("CausalLM" in architecture) or ("GPT2LMHeadModel" in architecture)
                assert self.encoder_decoder_model or self.decoder_only_model, f"Unknown HuggingFace model class: {model}"
                self.tokenizer = AutoTokenizer.from_pretrained(model if checkpoint is None else checkpoint,
                                                               trust_remote_code=True, **tok_kwargs)
                print("-------------- HF_HOME:", os.getenv("HF_HOME"))
                self.rationale = True
                AutoModelClass = AutoModelForSeq2SeqLM if self.encoder_decoder_model else AutoModelForCausalLM
                if checkpoint:
                    self.rationale = False
                    if self.device_map:
                        self.model = AutoModelClass.from_pretrained(checkpoint, device_map=self.device_map, **hf_kwargs)
                    else:
                        self.model = AutoModelClass.from_pretrained(checkpoint, **hf_kwargs).to(self.device)
                else:
                    if self.device_map:
                        self.model = AutoModelClass.from_pretrained(model, device_map=self.device_map, **hf_kwargs)
                    else:
                        self.model = AutoModelClass.from_pretrained(model, **hf_kwargs).to(self.device)
                self.drop_prompt_from_output = True
            except ValueError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model if checkpoint is None else checkpoint,
                    device_map=self.device_map, **hf_kwargs
                )
                self.drop_prompt_from_output = True
                self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, **tok_kwargs)
                self.drop_prompt_from_output = True
            if "llama" in self.model_name.lower():
                self.tokenizer.eos_token = '<|eot_id|>'
        self.history = []

    def _generate(self, prompt, **kwargs):
        assert not self.is_client
        # TODO: Add caching
        kwargs = {**openai_to_hf(**self.kwargs), **openai_to_hf(**kwargs)}
        # print(prompt)
        if isinstance(prompt, dict):
            try:
                prompt = prompt['messages'][0]['content']
            except (KeyError, IndexError, TypeError):
                print("Failed to extract 'content' from the prompt.")
        try:
            chat = dspy_to_chat(prompt)
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True)
        except Exception as e:
            logging.debug(f"Could not apply chat template for {self.model_name}: {e}")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)

        if self.drop_prompt_from_output:
            input_length = inputs.input_ids.shape[1]
            outputs = outputs[:, input_length:]

        decoding = [c for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        completions = [
            {
                "text": self.llama3_re.sub("", c).strip()
            }
            for c in decoding
        ]

        response = {
            "prompt": prompt,
            "choices": completions,
        }

        print("\n\n----- Model prompt: \n" + prompt)
        print("\n\n----- Model output: \n" + completions[0]["text"])
        return response
