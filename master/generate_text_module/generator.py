import json
import os
from typing import Dict, Any

from dotenv import load_dotenv
from langfuse import Langfuse, observe, get_client, propagate_attributes
from openai import OpenAI
import base64
import requests

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


def format_products_for_context(product_list):
    """Format product dictionaries as readable text for LLM context"""
    if not product_list:
        return []
    formatted_products = []
    for product in product_list:
        product_text = (
            f"Product ID: {product['id']}\n"
            f"Name: {product['name']}\n"
            f"Price: {product['price']}đ\n"
            f"Stock: {product['stock']}\n"
            f"Description: {product['description']}\n"
            f"Active: {'Yes' if product['is_active'] else 'No'}\n"
            f"---"
        )
        formatted_products.append(product_text)

    return "\n\n".join(formatted_products)

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

def build_message(context, prompt_input, image_link, chat_history):
    """
    Build OpenAI chat format messages from context and prompt.
    Returns a list of message dictionaries for OpenAI API.
    """
    system_content = (
        f"You are a helpful shop flower assistant, respectful and honest. "
        f"You are providing recommendation for the customer about different type of flower bouquets based on their given input. "
        f"Prioritize asking the customer for personal liking first, then give suggestion based on that. "
        f"Always give recommendation based on the flower products given as follow: {context} "
        f"If no flower products are given, give honest answer that no products match the user description"
        f"When the customer asks a general question like: Shop bạn bán những loại hoa nào or Bạn có thể liệt kê giúp tôi một số loại hoa, "
        f"you could: "
        f"- Provide a short list of representative flower products from the given context. "
        f"- Vary the list so answers don't feel repetitive. "
        f"- Encourage the customer to share their preferences (occasion, price range, style) so you can refine suggestions. "
        f"- Don't make up any of the information. "
        f"Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        f"Only name and price is necessary information, others should be cut out to shorten the answer"
        f"If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        f"List out the price for each of the product"
        f"Old chat history for reference: {chat_history}. "
        f"OPTIONAL: If exist products in recommendation list, using the markdown link syntax with this format: https://flowerplus.site//product/product_id where replace the product_id with actual product ID."
        f"Don't ask customer about combination of flower, only ask about complete product description or information"
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

    print(messages)
    return messages

def build_message_schema(schema_context, relational_context, prompt_input, top_k):
    """
    Build OpenAI chat format messages from context and prompt.
    Returns a list of message dictionaries for OpenAI API.
    """
    system_content = (
        f"You are a SQL assistant. Generate SQL queries based on the given schema, relationships user input."
        f"Follow the response format"
        f"product_type attribute must be set to 0 and is_custom must also be set to 0 if the product table is query"
        f"Don't select images, created_at, updated_at, product_string, sync_status when create the query script"
        f"Follow the response format of: product id, name, price, stock, description, is_active."
        f"If the user query ask about a specific occasion, for example birthday or christmas, be sure to join it with the sub table product_categories and the categories table and compare the product category"
        f"Most of the products in db contain extra 'Bó hoa' or 'Sản phẩm' in it name. So query for name must use 'like' rather than '='"
        f"Schema: {schema_context}"
        f"Relational context: {relational_context}"
        f"Use the following top_k recommendations: {top_k}"
    )

    # Initialize the user content list
    user_content = [{"type": "text", "text": prompt_input}]

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return messages

def build_message_intent(prompt_input):
    system_content = (
        f"You are a master at language trying to figure out what is the meaning of the user input."
        f"Translate the sentence into English for better analyzation, or keep the sentence in Vietnamese if you can't capture the meaning"
        f"The user is a flower shop customer"
        f"Analyze the user input thoroughly and guess the intend behind that."
        f"Your answer must be one of the following: "
        f"If the user input describe general information of the product or flower they want to buy or search. For example: tôi muốn mua một bó hoa màu xanh nhạt. Classify it as 'product_general'"
        f"If the user input describe specific shop products or shop flowers information, For example: price, product stock, specific occasion like birthdate or valentine, specific product name and product vouchers. Classify it as 'specific_information'"
        f"If the user input about shop information in general, For example: Shop hoa này đặt ở đâu ? or thời gian giao hàng là gì?. Classify it as 'shop_information'"
        f"If the user input aren't fall into any of the above. Classify it as 'general_input'"
        f"If the user ask about any past information, For example: còn sản phẩm nào tương tự vậy không. Classify it as 'product_general'"
        f"If the user mention about custom made flower or how to use the system. Classify it as 'shop_information'"
        f"Also try to find any clue of the user indicate the number of product he/she want to view"
        f"If no clue is found, default the top k value to 50"
    )

    # Initialize the user content list
    user_content = [{"type": "text", "text": prompt_input}]

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
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
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        print(f"OpenAI GPT-4o client initialized successfully with model: {self.model}")

    def _define_response_schema(self) -> Dict[str, Any]:
        """Define the response schema"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "sql_generation_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "can_generate_sql": {
                            "type": "boolean",
                            "description": "Whether SQL query can be generated"
                        },
                        "sql_query": {
                            "type": ["string", "null"],
                            "description": "Generated SQL query if can_generate_sql is true"
                        },
                    },
                    "required": ["can_generate_sql", "sql_query"],
                    "additionalProperties": False
                }
            }
        }

    def _define_intent_schema(self) -> Dict[str, Any]:
        """Define the response schema"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "sql_generation_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": ["string", "null"],
                            "description": "User intent with the following input"
                        },
                        "top_k": {
                            "type": ["integer", "null"],
                            "description": "The number of most likely product user want to view"
                        },
                    },
                    "required": ["intent", "top_k"],
                    "additionalProperties": False
                }
            }
        }

    def _define_payment_schema(self) -> Dict[str, Any]:
        """Define the response schema"""
        return {
            "type": "function",
            "function": {
                "name": "payment_url_generation",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "voucherCode": {
                            "type": "string",
                            "description": "voucher apply for the cart"
                        },
                        "userId": {
                            "type": "integer",
                            "description": "User id of the cart"
                        },
                        "productId": {
                            "type": "integer",
                            "description": "Product id of the cart"
                        },
                        "shippingAddress": {
                            "type": "string",
                            "description": "Shipping address of the cart"
                        },
                        "phoneNumber": {
                            "type": "string",
                            "description": "Phone number of the cart"
                        },
                        "recipientName": {
                            "type": "string",
                            "description": "Recipient name of the cart"
                        },
                        "note": {
                            "type": "string",
                            "description": "Note of the cart"
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Quantity of the product"
                        },
                        "orderId": {
                            "type": "integer",
                            "description": "Order id of the cart"
                        }
                    },
                    "required": ["userId", "productId", "shippingAddress", "phoneNumber", "quantity", "orderId"],
                    "additionalProperties": False
                }
            }
        }


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


    def generate_sql_query(self, messages):
        print(f"Generating answer with OpenAI {self.model}")

        try:
            # OpenAI API call with same generation settings as before
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Same as before
                top_p=0.95,  # Same as before (OpenAI uses top_p, not top_k)
                max_tokens=1024,  # Same as max_new_tokens before
                frequency_penalty=0.2,  # Similar effect to repetition_penalty=1.2
                response_format=self._define_response_schema()
            )

            output = response.choices[0].message.content
            result = json.loads(output)
            print(f"Generated answer: {output}...")  # Print first 100 chars

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"Error generating answer with OpenAI: {error_msg}")
            raise  # Re-raise the exception

    def generate_intent_query(self, messages):
        try:
            # OpenAI API call with same generation settings as before
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Same as before
                top_p=0.95,  # Same as before (OpenAI uses top_p, not top_k)
                max_tokens=1024,  # Same as max_new_tokens before
                frequency_penalty=0.2,  # Similar effect to repetition_penalty=1.2
                response_format=self._define_intent_schema()
            )

            output = response.choices[0].message.content
            result = json.loads(output)
            print(f"Generated answer: {output}...")  # Print first 100 chars
            # If output is empty or too short, return a default message

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"Error generating answer with OpenAI: {error_msg}")
            raise  # Re-raise the exception








