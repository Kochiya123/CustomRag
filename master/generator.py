import torch
from transformers import AutoTokenizer, pipeline

llm_name = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=llm_name,
    torch_dtype = torch.float16,
    device_map = 'cuda',
)