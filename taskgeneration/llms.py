import os
import time
import anthropic
import openai
import numpy as np
from dotenv import load_dotenv
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
import ipdb

def get_llm(engine, temp=0.0, max_tokens=1):
    '''
    Based on the engine name, returns the corresponding LLM object
    '''
    Q_, A_ = '\n\nQ:', '\n\nA:'
    if engine.startswith("llama-3"):
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

        if engine.startswith('llama-2'):
            # engine = 'meta-llama/Llama-2-8b-chat-hf'
            engine = 'meta-llama/Llama-2-70b-chat-hf' 
            self.tokenizer = AutoTokenizer.from_pretrained(
                engine, use_auth_token=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                engine, device_map="auto", torch_dtype=torch.bfloat16, use_auth_token=True)

        elif engine.startswith('llama-3'):
            # engine = "meta-llama/Meta-Llama-3-8B-Instruct"
            engine = "meta-llama/Meta-Llama-3-70B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(engine, use_auth_token=True)
            self.model = AutoModelForCausalLM.from_pretrained(engine, device_map="auto", torch_dtype=torch.float16, use_auth_token=True)#torch.bfloat16
            # self.pipe = pipeline(
            #     "text-generation",
            #     model=engine,
            #     model_kwargs={"torch_dtype": torch.bfloat16},
            #     device="cuda",
            #     device_map="auto",
            # )

        else:
            print("Wrong engine name for HF API LLM")
            raise NotImplementedError
        
        self.instruct = True if "Instruct" in engine else False
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.max_tokens = max_tokens
        self.temperature = temperature + 1e-6

    def generate(self, text, temp=0.0, max_tokens=1, arms=None):
       
        ## chat template-based text generation
        outputs = self.model.generate(
                        text,
                        max_new_tokens=self.max_tokens,
                        eos_token_id=self.terminators,
                        do_sample=True,
                        temperature=self.temperature,
                        # top_p=0.9,
                    )
        response = self.tokenizer.decode(outputs[0][text.shape[-1]:], skip_special_tokens=True)

        ## vanilla text generation
        # model_inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        # model_outputs = self.model.generate(
        #     **model_inputs, max_new_tokens=self.max_tokens, temperature=self.temperature)
        # response = self.tokenizer.decode(
        #     model_outputs[0][model_inputs.input_ids[0].shape[0]:])

        ## pipeline-based text generation
        # response = self.pipe(
        #     text,
        #     max_new_tokens=self.max_tokens,#8000
        #     eos_token_id=self.terminators,
        #     do_sample=True,
        #     temperature=self.temperature, #0.6
        #     #top_p=0.9,
        # )[0]["generated_text"][-1]["content"]

        return response
