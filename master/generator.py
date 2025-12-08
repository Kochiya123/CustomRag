import os
from dotenv import load_dotenv
import torch
from langfuse import Langfuse, observe, get_client, propagate_attributes
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, GenerationConfig
from transformers import BitsAndBytesConfig

load_dotenv()

#create langfuse for logging and tracing llm call
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    # optional if self-hosted
    # base_url="http://localhost:8000"
)

def build_context(products):
    context = "\n".join([
        f"- {p[0]} (${p[2]}): {p[1]}"
        for p in products
    ])
    return context


def build_message(context, prompt_input):
    system_instructions = (
        f"<s>[INST] <<SYS>> You are a helpful shop flower assistant, respectful and honest assistant. \
        You are providing recommendation for the customer about different type of flowers based on their given input \
        Prioritize asking the customer for personal liking first, then give suggestion based on that.\
        Your answer must be based on the flower products provided as follow:  {context} \
        When the customer asks a general question like : Shop bạn bán những loại hoa nào or Bạn có thể liệt kê giúp tôi một số loại hoa,\
        you should:\
        - Provide a short list (3–5) of representative flower products from the given context.\
        - Vary the list so answers don’t feel repetitive.\
        - Encourage the customer to share their preferences (occasion, price range, style) so you can refine suggestions.\
        - Always base your answer only on the provided product context.\
        - Keep the tone friendly, professional, and helpful.\
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\
        Please ensure that your responses are socially unbiased and positive in nature.\
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
        If you don't know the answer to a question, please response as language model you are not able to respone detailed to these kind of question.\n<</SYS>>\n\n {prompt_input} [/INST] "
    )

    return system_instructions

def build_message_general(prompt_input):
    system_instructions = (
        f"<s>[INST] <<SYS>> You are a helpful, respectful, and honest assistant. \
        You provide clear, concise, and informative responses to the customer based on their input. \
        Keep the tone friendly, professional, and helpful. \
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. \
        Please ensure that your responses are socially unbiased and positive in nature. \
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
        If you don't know the answer to a question, respond that as a language model you are not able to provide detailed information on that topic.\n<</SYS>>\n\n {prompt_input} [/INST] "
    )

    return system_instructions

model_name = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
local_dir = "../models/transformers/"

@observe
class Generator_llm():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure 8-bit quantization using BitsAndBytesConfig (new way)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        self.m = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            pretraining_tp=1,
            # use_auth_token=True,
            # trust_remote_code=True,
            cache_dir=local_dir,
            device_map=self.device,
            revision="main"
        )

        self.tok = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_dir,
            padding_side="right",
            use_fast=False,
            tokenizer_type='llama',
            token=True,  # Replaced deprecated use_auth_token
        )
        self.tok.bos_token_id = 1

        print("Model loaded successfully")

    def generate_answer(self, messages, user_session_id, user_id):
        with propagate_attributes(session_id=user_session_id, user_id=user_id):
            # Tokenize the input messages
            inputs = self.tok(messages, return_tensors="pt")
            print(len(inputs))

            stop = StopOnTokens()
            print("Generating answer")
            # Generation configuration
            generation_config = {
                "temperature": 0.2,
                "top_k": 30,
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.2,
                "max_new_tokens": 800,
                "early_stopping": True,
                "stopping_criteria": StoppingCriteriaList([stop]),
            }

            with torch.no_grad():
                generation_output = self.m.generate(
                    input_ids = inputs["input_ids"].to(self.device),
                    attention_mask = inputs['attention_mask'].to(self.device),
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                    **generation_config
                )

            s = generation_output[0]
            output = self.tok.decode(s, skip_special_tokens=True)
            print(output)
            output = output.split("[/INST]", 1)[-1].strip()

            langfuse = get_client()
            langfuse.flush()
            return output

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Stop on EOS token or padding token
        # Token ID 0 might be too aggressive, so we'll rely on eos_token_id instead
        stop_token_ids = []
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False