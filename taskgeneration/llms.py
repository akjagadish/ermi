import os
import time
import anthropic
import openai
import numpy as np
from dotenv import load_dotenv
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_llm(engine, temp=0.0, max_tokens=1):
    '''
    Based on the engine name, returns the corresponding LLM object
    '''
    Q_, A_ = '\n\nQ:', '\n\nA:'
    if engine.startswith("llama-2"):
        load_dotenv()
        hf_key = os.getenv(f"HF_API_KEY")
        llm = HF_API_LLM((hf_key, engine, max_tokens, temp))
    else:
        print('No key found')
        raise NotImplementedError
    return llm, Q_, A_


class LLM:
    def __init__(self, llm_info):
        self.llm_info = llm_info

    def generate(self, text, temp=0.0, max_tokens=1, arms=None):
        raise NotImplementedError


class HF_API_LLM(LLM):
    def __init__(self, llm_info):
        hf_key, engine, max_tokens, temperature = llm_info
        padtokenId = 50256  # Falcon needs that to avoid some annoying warning
        # Authenticate
        from huggingface_hub import notebook_login
        notebook_login(hf_key)

        if engine.startswith('llama-2'):
            # # Change llama-2-* to meta-llama/Llama-2-*b-hf
            # if 'chat' in engine:
            #     engine = 'meta-llama/L' + \
            #         engine[1:].replace('-chat', '') + 'b-chat-hf'
            # else:
            #     engine = 'meta-llama/L' + engine[1:] + 'b-hf'
            engine = 'meta-llama/Llama-2-7b-chat-hf'
        else:
            print("Wrong engine name for HF API LLM")
            raise NotImplementedError

        print(engine)
        self.tokenizer = AutoTokenizer.from_pretrained(
            engine, use_auth_token=hf_key)
        self.model = AutoModelForCausalLM.from_pretrained(
            engine, device_map="auto", torch_dtype=torch.bfloat16, use_auth_token=hf_key)
        self.max_tokens = max_tokens
        # Adapt pipeline to set temperature to 0
        self.temperature = temperature + 1e-6

    def generate(self, text, temp=0.0, max_tokens=1, arms=None):
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        model_outputs = self.model.generate(
            **model_inputs, max_new_tokens=self.max_tokens, temperature=self.temperature)
        response = self.tokenizer.decode(
            model_outputs[0][model_inputs.input_ids[0].shape[0]:])
        return response
