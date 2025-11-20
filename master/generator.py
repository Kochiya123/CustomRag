import numpy as np
import psycopg2
from langchain_classic.chains import create_retrieval_chain
import torch
from transformers import AutoTokenizer, pipeline

class Generator_llm:
    def __init__(self):
        self.llm_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.pipe = pipeline(
            "text-generation",
            model=self.llm_name,
            torch_dtype=torch.float16,
            device_map='cuda',
        )

def build_context(products):
    context = "\n".join([
        f"- {p[0]} (${p[2]}): {p[1]}"
        for p in products
    ])
    return context

def build_messages(context, query):
    messages = [
        {"role": "system", "content": "You are a helpful flower shop assistant."
                    "Provide one Answer ONLY the following query based on the context provided below. "
                    "Do not generate or answer any other questions. "
                    "Do not make up or infer any information that is not directly stated in the context. "
                    "Provide a concise answer."
                    f"{context}"},
        {"role": "user", "content": query},
    ]
    return messages

def generate_reponse(self, messages):
    response = self.pipe(messages, max_new_tokens=128)[-1]["generated_text"][-1]["content"]
    return response