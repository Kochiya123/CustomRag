import os
from dotenv import load_dotenv
from langfuse import Langfuse, observe, get_client, propagate_attributes
from openai import OpenAI
from openai.types.beta.threads import image_url
import base64
import requests
from io import BytesIO

load_dotenv()

#create langfuse for logging and tracing llm call
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    # optional if self-hosted
    # base_url="http://localhost:8000"
)

def build_context(products):
    """
    Build context string from product list.
    Products format: (product_id, product_name, product_text, price)
    or legacy format: (product_name, product_text, price) for backward compatibility
    """
    context_lines = []
    for p in products:
            # Legacy format: (product_name, product_text, price)
            product_id, product_name, product_text, price = p
            context_lines.append(f"- {product_name} (${price}): {product_text}")
    return "\n".join(context_lines)

def build_category_context(categories):
    """
    Build context string from category list.
    Categories format: (category_id, category_name)
    """
    context = "\n".join([
        f"- {cat[1]} (ID: {cat[0]})"
        for cat in categories
    ])
    return context


def encode_image_to_base64(image_source):
    """
    Convert image to base64 string.
    image_source can be a URL or file path.
    """
    if image_source.startswith('http'):
        # Download from URL
        response = requests.get(image_source)
        image_data = response.content
    else:
        # Read from file
        with open(image_source, 'rb') as image_file:
            image_data = image_file.read()

    return base64.b64encode(image_data).decode('utf-8')

def build_message(context, prompt_input, image_link):
    """
    Build OpenAI chat format messages from context and prompt.
    Returns a list of message dictionaries for OpenAI API.
    """
    system_content = (
        f"You are a helpful shop flower assistant, respectful and honest. "
        f"You are providing recommendation for the customer about different type of flower bouquets based on their given input. "
        f"Prioritize asking the customer for personal liking first, then give suggestion based on that. "
        f"Your answer must be based on the flower products provided as follow: {context} "
        f"If an image link is provided via user query, process the image if able or else answer based on the context without checking the image."
        f"Removed any unnecessary or unnatural information or  included in context. "
        f"When the customer asks a general question like: Shop bạn bán những loại hoa nào or Bạn có thể liệt kê giúp tôi một số loại hoa, "
        f"you should: "
        f"- Provide a short list of representative flower products from the given context. "
        f"- Vary the list so answers don't feel repetitive. "
        f"- Encourage the customer to share their preferences (occasion, price range, style) so you can refine suggestions. "
        f"- Always base your answer only on the provided product: {context}. "
        f"- Don't make up any of the information. "
        f"- Keep the tone friendly, professional, and helpful. "
        f"Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        f"Don't leave any uncommon tag or mark in the answer"
        f"Please ensure that your responses are socially unbiased and positive in nature. "
        f"If a question does not make any sense, or is not factually coherent, or no context is provided, explain why instead of answering something not correct. "
        f"If you don't know the answer to a question, please response as language model you are not able to respone detailed to these kind of question."
    )

    # Initialize the user content list
    user_content = [{"type": "text", "text": prompt_input}]

    # If an image link is provided, append the image_url object
    if image_link:
        try:
            # Get base64 encoded image
            base64_image = encode_image_to_base64(image_link)

            # Determine image format
            image_format = "jpeg"  # default
            if image_link.lower().endswith('.png'):
                image_format = "png"
            elif image_link.lower().endswith('.webp'):
                image_format = "webp"
            elif image_link.lower().endswith('.gif'):
                image_format = "gif"

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{base64_image}"
                }
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            # Continue without image

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    return messages

def build_message_general(prompt_input):
    """
    Build OpenAI chat format messages for general questions.
    Returns a list of message dictionaries for OpenAI API.
    """
    system_content = (
        f"You are a helpful, respectful, and honest assistant. "
        f"You provide clear, concise, and informative responses to the customer based on their input. "
        f"Keep the tone friendly, professional, and helpful. "
        f"Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        f"Please ensure that your responses are socially unbiased and positive in nature. "
        f"If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        f"If you don't know the answer to a question, respond that as a language model you are not able to provide detailed information on that topic."
    )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt_input}
    ]
    
    return messages


class Generator_llm():
    def __init__(self):
        api_key = os.getenv("OPEN_AI_TOKEN")
        if not api_key:
            raise ValueError("OPEN_AI_TOKEN environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        # Use GPT-4o (latest GPT-4 model)
        # Can be overridden via OPENAI_MODEL environment variable
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        print(f"OpenAI GPT-4o client initialized successfully with model: {self.model}")

    @observe
    def generate_answer(self, messages, user_session_id, user_id):
        """
        Generate answer using OpenAI GPT-4o API.
        
        Args:
            messages: List of message dictionaries (from build_message or build_message_general)
            user_session_id: Session ID for Langfuse tracking
            user_id: User ID for Langfuse tracking
        
        Returns:
            str: Generated answer text
        """
        with propagate_attributes(session_id=user_session_id, user_id=user_id):
            print(f"Generating answer with OpenAI {self.model}")
            
            try:
                # OpenAI API call with same generation settings as before
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,  # Same as before
                    top_p=0.95,  # Same as before (OpenAI uses top_p, not top_k)
                    max_tokens=1024,  # Same as max_new_tokens before
                    # Note: OpenAI doesn't support repetition_penalty directly,
                    # but frequency_penalty can help reduce repetition
                    frequency_penalty=0.2,  # Similar effect to repetition_penalty=1.2
                )
                
                output = response.choices[0].message.content.strip()
                print(f"Generated answer: {output[:100]}...")  # Print first 100 chars
                
                # If output is empty or too short, return a default message
                if not output or len(output) < 3:
                    print("Warning: Generated output is empty or too short")
                    return "Xin lỗi, tôi không thể tạo câu trả lời. Vui lòng thử lại."
                
                langfuse_client = get_client()
                langfuse_client.flush()
                return output
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error generating answer with OpenAI: {error_msg}")
                langfuse_client = get_client()
                langfuse_client.flush()
                raise  # Re-raise the exception