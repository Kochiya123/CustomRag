import os

import numpy as np
import psycopg2
from huggingface_hub import login
from langchain_classic.chains import create_retrieval_chain
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer


def build_context(products):
    context = "\n".join([
        f"- {p[0]} (${p[2]}): {p[1]}"
        for p in products
    ])
    return context


def build_message(context, query):
    system_instructions = (
        "You are a helpful flower shop assistant. "
        "Provide one Answer ONLY to the following query based on the context provided below in Vietnamese. "
        "Answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Do not make up or infer any information that is not directly stated in the context. "
        "Provide a concise answer. "
        "If you don't know the answer to a question, please respond that you cannot provide a detailed answer.\n"
    )

    system_content = f"{system_instructions}\nContext:\n{context}"

    # Build full prompt
    full_prompt = f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{query} [/INST]"
    return full_prompt

model_name = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
local_dir = "../models/transformers/"

class Generator_llm():
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## Loading Base LLaMa model weight and Merge with Adapter Weight with the base model
        self.m = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            pretraining_tp=1,
            # use_auth_token=True,
            # trust_remote_code=True,
            cache_dir=local_dir,
        )

        self.tok = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_dir,
            padding_side="right",
            use_fast=False,
        )
        self.tok.bos_token_id = 1


        generation_config = dict(
            temperature=0.2,
            top_k=20,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.2,
            max_new_tokens=200,
            early_stopping=True,
        )
        print("model loaded")

    def generate_answer(self, messages):
        input_ids = self.tok(messages, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.m.device)
        stop = StopOnTokens()
        streamer = TextIteratorStreamer(self.tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    
        # #print(tok.decode(output[0]))
        generation_config = dict(
            temperature=0.1,
            top_k=30,
            top_p=0.95,
            do_sample=True,
            # num_beams=1,
            repetition_penalty=1.2,
            max_new_tokens=2048, ## 8K
            early_stopping=True,
            stopping_criteria=StoppingCriteriaList([stop]),
        )
        inputs = self.tok(messages,return_tensors="pt")  #add_special_tokens=False ?
        generation_output = self.m.generate(
            input_ids = inputs["input_ids"].to(self.device),
            attention_mask = inputs['attention_mask'].to(self.device),
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
            **generation_config
        )
        generation_output_ = self.m.generate(input_ids = inputs["input_ids"].to(self.device), **generation_config)
    
        s = generation_output[0]
        output = self.tok.decode(s,skip_special_tokens=True)
        #response = output.split("### Output:")[1].strip()
        return output


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_token_ids = [0]
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False