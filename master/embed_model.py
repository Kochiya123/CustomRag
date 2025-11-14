from transformers import AutoModel
import torch
from huggingface_hub import notebook_login

hf_token = "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"
notebook_login()

model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True, dtype = torch.float16)

model.to("cuda" if torch.cuda.is_available() else "cpu")

