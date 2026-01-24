from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_restful import Api
from flasgger import Swagger
import os
import tiktoken

from langfuse import Langfuse

from master.db_module.connect import connect
from master.db_service.postgre_service import MysqlService
from master.generate_text_module.generator import Generator_llm, build_message, build_message_schema, \
    build_message_intent, format_products_for_context
from master.embed_module.embed_llm import Embed_llm, save_chat_history, load_chat_history, \
    get_chat_history_user_session, get_latest_history_user_session, reset_session_method

from master.guardrail.guardrail import Guardrail
from master.recommendation_service.rec_service import HybridRecommender

from master.rerank_module.rerank_llm import Rerank

load_dotenv()

#create langfuse for logging and tracing llm call
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    # optional if self-hosted
    # base_url="http://localhost:8000"
)
cur, conn = connect()


generator = Generator_llm()
embed = Embed_llm()
guard = Guardrail()
reranker = Rerank()

def get_recommender(conn):
    """Initialize recommender with database connection"""
    if conn is None:
        raise Exception("Failed to connect to database")
    return HybridRecommender(conn)

def count_tokens(text, model="gpt-4o"):
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback: rough estimation (1 token ≈ 4 characters for English, but Vietnamese might be different)
        return len(text) // 3

def _generate_reason(recommendation: dict) -> str:
    """Generate a human-readable reason for the recommendation"""
    score = recommendation.get('score', 0)

    if score > 0.8:
        return "Highly recommended based on your interests"
    elif score > 0.6:
        return "Popular among users with similar taste"
    elif score > 0.4:
        return "You might also like this"
    else:
        return "Trending in your favorite categories"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Allow non-ASCII characters in JSON responses (for Vietnamese)
api = Api(app)

# Initialize Swagger
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "RAG API",
        "description": "API for RAG (Retrieval-Augmented Generation) system",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": [
        "http",
        "https"
    ]
})

@app.route('/product', methods=['POST'])
def embed_single():
    """
    Embed single product
    ---
    tags:
      - Products
    summary: Embed a single product
    description: This endpoint embeds a single product
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - product_string
            - product_id
            - category_id
            - product_name
            - price
          properties:
            product_string:
              type: string
              description: The combined string description of the product to embed
              example: "Hoa màu vàng, tươi mới, phù hợp cho ngày của mẹ"
            product_id:
              type: integer
              description: Unique identifier for the product
              example: 123
            category_id:
              type: integer
              description: Category ID that this product belongs to
              example: 5
            product_name:
              type: string
              description: Name of the product
              example: "Hoa hướng dương"
            price:
              type: number
              description: Price of the product
              example: 150000
    responses:
      200:
        description: Embed single product successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: Embed 123 product successfully
      400:
        description: Missing required fields
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Missing required fields: product_string, product_id, category_id, product_name, price"
      500:
        description: Embed single product failed
        schema:
          type: object
          properties:
            message:
              type: string
              example: Embed single product failed
    """
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Request body must be JSON'}), 400
    
    product_string = data.get('product_string')
    product_id = data.get('product_id')
    category_id = data.get('category_id')
    product_name = data.get('product_name')
    price = data.get('price')

    
    if not product_string or not product_id or not category_id or not product_name or not price:
        return jsonify({'message': 'Missing required fields: product_string, product_id, category_id, product_name, price'}), 400
    
    result = embed.embedded_add_single_column_product(cur, conn, product_string, product_id, category_id)
    if result:
        return jsonify({'message': f'Embed product {result} successfully'}), 200
    return jsonify({'message': 'Embed single product failed'}), 500

@app.route('/product/<product_id>', methods=['PUT'])
def embed_update_single(product_id):
    """
    Update the embed of a single product
    ---
    tags:
      - Products
    summary: Update the embed of a single product
    description: This endpoint Update the embed of a single product by its ID
    parameters:
      - name: product_id
        in: path
        type: string
        required: true
        description: The ID of the product to embed
        example: "123"
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - product_string
            - product_id
            - category_id
            - product_name
            - price
          properties:
            product_string:
              type: string
              description: The combined string description of the product to embed
              example: "Hoa màu vàng, tươi mới, phù hợp cho ngày của mẹ"
            product_id:
              type: integer
              description: Unique identifier for the product
              example: 123
            category_id:
              type: integer
              description: Category ID that this product belongs to
              example: 5
            product_name:
              type: string
              description: Name of the product
              example: "Hoa hướng dương"
            price:
              type: number
              description: Price of the product
              example: 150000
    responses:
      200:
        description: Update the embed successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: Update embed 123 product successfully
      400:
        description: Missing required fields
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Missing required fields: context, image_url"
      500:
        description: Update the embed failed
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Update embed 123 product failed"
    """
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Request body must be JSON'}), 400
    
    product_string = data.get('product_string')
    product_id = data.get('product_id')
    category_id = data.get('category_id')
    product_name = data.get('product_name')
    price = data.get('price')
    
    if not product_string or not product_id or not category_id or not product_name or not price:
        return jsonify(
            {'message': 'Missing required fields: product_string, product_id, category_id, product_name, price'}), 400
    
    result = embed.embedded_update_single_column_product(cur, conn, product_id, product_string, category_id)
    if result:
        return jsonify({'message': f'Update embed product {product_id} successfully'}), 200
    return jsonify({'message': f'Update embed product {product_id} failed'}), 500

@app.route('/product/<product_id>', methods=['DELETE'])
def embed_delete_single(product_id):
    """
    Delete single product
    ---
    tags:
      - Products
    summary: Delete a single product
    description: This endpoint deletes a single product and removes its embedding from the database
    parameters:
      - name: product_id
        in: path
        type: integer
        required: true
        description: The unique identifier of the product to delete
        example: 123
    responses:
      200:
        description: Delete single product successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: Delete 123 product successfully
      400:
        description: Missing required fields
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Missing required fields: product_id"
      500:
        description: Delete single product failed
        schema:
          type: object
          properties:
            message:
              type: string
              example: Delete single product failed
    """
    if not product_id:
        return jsonify({'message': 'Missing required fields: product_id'}), 400

    result = embed.embedded_delete_single_column_product(cur, conn, product_id)
    if result == product_id:
        return jsonify({'message': f'Delete product {product_id} successfully'}), 200
    return jsonify({'message': f'Delete single product {product_id} failed: '}), 500

@app.route('/category', methods=['POST'])
def embed_single_category():
    """
    Add a new category with embedding
    ---
    tags:
      - Categories
    summary: Add a new category and generate its embedding
    description: |
      This endpoint creates a new category and generates a vector embedding for it.
      The embedding is used for semantic search and retrieval.
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - category_id
            - category_name
          properties:
            category_id:
              type: integer
              description: Unique identifier for the category
              example: 1
            category_name:
              type: string
              description: Name of the category
              example: "Hoa tươi"
    responses:
      200:
        description: Category added successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Add category 1 successfully"
      400:
        description: Missing required fields or invalid request
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Missing required fields: category_id, category_name"
      500:
        description: Failed to add category
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Add category 1 failed"
    """
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Request body must be JSON'}), 400

    category_id = data.get('category_id')
    category_name = data.get('category_name')
    if not category_id or not category_name:
        return jsonify({'message': 'Missing required fields: category_id, category_name'}), 400

    result = embed.embedded_add_single_category(cur, conn, category_id, category_name)
    if result:
        return jsonify({'message': f'Add category {category_id} successfully'}), 200
    return jsonify({'message': f'Add category {category_id} failed'}), 500

@app.route('/category/<category_id>', methods=['PUT'])
def embed_update_single_category(category_id):
    """
    Update a category and its embedding
    ---
    tags:
      - Categories
    summary: Update an existing category and regenerate its embedding
    description: |
      This endpoint updates a category's information and regenerates its vector embedding.
    parameters:
      - name: category_id
        in: path
        type: integer
        required: true
        description: The unique identifier of the category to update
        example: 1
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - category_id
            - category_name
          properties:
            category_id:
              type: integer
              description: Unique identifier for the category
              example: 1
            category_name:
              type: string
              description: Updated name of the category
              example: "Hoa tươi cao cấp"
    responses:
      200:
        description: Category updated successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Update category 1 successfully"
      400:
        description: Missing required fields or invalid request
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Missing required fields: category_id, category_name"
      500:
        description: Failed to update category
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Update category 1 failed"
    """
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Request body must be JSON'}), 400

    category_id = data.get('category_id')
    category_name = data.get('category_name')
    if not category_id or not category_name:
        return jsonify({'message': 'Missing required fields: category_id, category_name'}), 400

    result = embed.embedded_update_single_category(cur, conn, category_id, category_name)
    if result:
        return jsonify({'message': f'Update category {category_id} successfully'}), 200
    return jsonify({'message': f'Update category {category_id} failed'}), 500

@app.route('/category/<category_id>', methods=['DELETE'])
def embed_delete_category(category_id):
    """
    Delete a category
    ---
    tags:
      - Categories
    summary: Delete a category and its embedding
    description: |
      This endpoint deletes a category and removes its vector embedding from the database.
    parameters:
      - name: category_id
        in: path
        type: integer
        required: true
        description: The unique identifier of the category to delete
        example: 1
    responses:
      200:
        description: Category deleted successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Delete category 1 successfully"
      500:
        description: Failed to delete category
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Delete category 1 failed"
    """
    result = embed.embedded_delete_single_column_category(cur, conn, category_id)
    if result:
        return jsonify({'message': f'Delete category {category_id} successfully'}), 200
    return jsonify({'message': f'Delete category {category_id} failed'}), 500

@app.route('/general', methods=['POST'])
def embed_general():
    """
    Add general information with embedding
    ---
    tags:
      - General Information
    summary: Add general information and generate its embedding
    description: |
      This endpoint adds general information and generates a vector embedding for semantic search.
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - general_text
          properties:
            general_text:
              type: string
              description: General information text to embed
              example: "Thông tin chung về cửa hàng hoa"
    responses:
      200:
        description: General information added successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Add general information successfully"
      400:
        description: Missing required fields
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Missing required fields: general_text"
      500:
        description: Failed to add general information
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Add general information failed"
    """
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Request body must be JSON'}), 400

    general_text = data.get('general_text')
    if not general_text:
        return jsonify({'message': 'Missing required fields: general_text'}), 400

    result = embed.embedded_add_general_information(cur, conn, general_text)
    if result:
        return jsonify({'message': f'Add general information successfully'}), 200
    return jsonify({'message': f'Add general information failed'}), 500

@app.route('/general/<general_id>', methods=['PUT'])
def embed_update_general_information(general_id):
    """
    Update general information and its embedding
    ---
    tags:
      - General Information
    summary: Update existing general information and regenerate its embedding
    description: |
      This endpoint updates general information and regenerates its vector embedding.
    parameters:
      - name: general_id
        in: path
        type: integer
        required: true
        description: The unique identifier of the general information to update
        example: 1
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - general_text
          properties:
            general_text:
              type: string
              description: Updated general information text
              example: "Thông tin cập nhật về cửa hàng hoa"
    responses:
      200:
        description: General information updated successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Update general information successfully"
      400:
        description: Missing required fields
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Missing required fields: general_text"
      500:
        description: Failed to update general information
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Update general information failed"
    """
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Request body must be JSON'}), 400
    general_text = data.get('general_text')
    if not general_text:
        return jsonify({'message': 'Missing required fields: general_text'}), 400
    result = embed.embedded_update_general_information(cur, conn, general_id, general_text)
    if result:
        return jsonify({'message': f'Update general information successfully'}), 200
    return jsonify({'message': f'Update general information failed'}), 500

@app.route('/general/<general_id>', methods=['DELETE'])
def embed_delete_general_information(general_id):
    """
    Delete general information
    ---
    tags:
      - General Information
    summary: Delete general information and its embedding
    description: |
      This endpoint deletes general information and removes its vector embedding from the database.
    parameters:
      - name: general_id
        in: path
        type: integer
        required: true
        description: The unique identifier of the general information to delete
        example: 1
    responses:
      200:
        description: General information deleted successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Delete general information successfully"
      500:
        description: Failed to delete general information
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Delete general information failed"
    """
    result = embed.embedded_delete_general_information(cur, conn, general_id)
    if result:
        return jsonify({'message': f'Delete general information successfully'}), 200
    return jsonify({'message': f'Delete general information failed'}), 500



@app.route('/', methods=['POST'])
def get_answer():
    """
    Generate answer for a query
    ---
    tags:
      - Query
    summary: Get answer for a query
    description: |
      This endpoint generates an answer for a given Vietnamese query using RAG (Retrieval-Augmented Generation).
      The system uses Jina Embeddings v4 for text and image encoding, Jina Rerank v3 for result reranking,
      and OpenAI GPT-4o for answer generation.
      
      The query can be about:
      - Specific flower products (with optional image URL for visual search)
      - Product categories
      - General shop information (delivery, policies, etc.)
      - General questions about flowers
      
      All queries are processed through embedding-based retrieval. If an image_url is provided,
      the system combines text-to-text and image-to-text similarity scores before reranking.

    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - query
          properties:
            query:
              type: string
              description: |
                The user query in Vietnamese. This is the question or request that the user wants answered.
                Maximum length: 600 tokens. The system will search for relevant products using vector embeddings
                and generate an answer based on the retrieved context.
              example: "tôi muốn mua hoa màu vàng cho ngày của mẹ"
            user_id:
              type: string
              required: false
              description: |
                Optional user identifier for tracking and analytics purposes.
                Used for Langfuse tracing to associate queries with specific users.
                If provided, the chat history will be saved to the database.
              example: "user_12345"
            session_id:
              type: string
              required: false
              description: |
                Optional session identifier for tracking conversation sessions.
                Used for Langfuse tracing to group related queries in a session.
              example: "session_abc123"
            image_url:
              type: string
              required: false
              description: |
                Optional image URL for image-based product search.
                If provided, the system will:
                1. Encode the image using Jina Embeddings v4
                2. Compute cosine similarity between image embedding and product text embeddings
                3. Combine image similarity scores with text-to-text similarity scores
                4. Rerank the combined results using Jina Rerank v3
                5. Use the reranked results as context for answer generation
              example: "https://cdn.pixabay.com/photo/2023/03/14/11/19/flower-7852094_1280.jpg"
            reset_session:
              type: boolean
              required: false
              default: false
              description: |
                Optional flag to reset the user's session. When set to true, the system will:
                1. Clear the existing chat history for the current session_id
                2. Start a new conversation session
                3. Generate a new session_id if one was not provided
                This is useful when the user wants to start a completely new conversation
                without the context of previous messages.
              example: true
        example:
          query: "tôi muốn mua hoa màu vàng cho ngày của mẹ"
          user_id: "user_12345"
          session_id: "session_abc123"
          image_url: "https://cdn.pixabay.com/photo/2023/03/14/11/19/flower-7852094_1280.jpg"
          reset_session: false

    responses:
      200:
        description: Successful answer generation
        schema:
          type: object
          properties:
            message:
              type: string
              description: The generated answer in Vietnamese based on the retrieved product context
              example: "Dựa trên các sản phẩm có sẵn, chúng tôi có các loại hoa màu vàng phù hợp cho ngày của mẹ..."
        examples:
          application/json:
            message: "Chúng tôi có nhiều loại hoa màu vàng đẹp cho ngày của mẹ, bao gồm hoa hướng dương, hoa cúc vàng, và hoa hồng vàng..."
      
      400:
        description: Bad request - Missing query in body, request body must be JSON, query too long (exceeds 600 tokens), or inappropriate content detected
        schema:
          type: object
          properties:
            message:
              type: string
              description: Error message explaining what went wrong
        examples:
          application/json:
            message: "Request body must be JSON"
      
      404:
        description: Product or category not found
        schema:
          type: object
          properties:
            message:
              type: string
              description: Message indicating no relevant products or categories were found
        examples:
          application/json:
            message: "Chúng tôi không tìm thấy sản phẩm nào cho danh mục Hoa sinh nhật!"
      
      500:
        description: Internal server error - System error or inappropriate response generated
        schema:
          type: object
          properties:
            message:
              type: string
              description: Error message indicating a system error occurred
        examples:
          application/json:
            message: "Hệ thống gặp trục trặc, vui lòng thử lại sau!"

    produces:
      - application/json
    consumes:
      - application/json
    """
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Request body must be JSON'}), 400
    query = data.get('query')
    user_id = data.get("user_id") or None
    image_url = data.get("image_url") or None
    session_id = data.get("session_id") or None

    if not query:
        return jsonify({'message': 'Missing query parameter "query"'}), 400

    # Check token limit (600 tokens max)
    query_tokens = count_tokens(query)
    if query_tokens > 600:
        return jsonify({
            'message': f'Query quá dài. Giới hạn là 600 tokens, nhưng query của bạn có {query_tokens} tokens. Vui lòng rút ngắn câu hỏi.'
        }), 400

    # Content moderation check on input
    if guard.guard_check__response(query):
        return jsonify({'message': 'Câu hỏi chứa nội dung không phù hợp'}), 400
    mysql = MysqlService()

    chat_history = get_latest_history_user_session(cur, conn, user_id)
    intent_message = build_message_intent(query)
    result = generator.generate_intent_query(intent_message)
    intent = result["intent"]
    top_k = result["top_k"]
    context = []
    messages = None

    if intent == "product_general":
        # vector search
        result = embed.retrieval_vector_product(cur, conn, query, top_k)
        result.sort(key=lambda x: x[1], reverse=True)
        product_list = mysql.get_product_on_id(result)
        context = format_products_for_context(product_list)
        messages = build_message(context, query, image_url, chat_history)

    elif intent == "specific_information":
        # sql query transformation
        table_schema = mysql.get_tables_schema()
        relation_schema = mysql.get_relational_schema()
        sql_message = build_message_schema(table_schema, relation_schema, query, top_k)
        result = generator.generate_sql_query(sql_message)
        sql_query = result["sql_query"]
        result = mysql.get_product_on_query(sql_query)
        context = format_products_for_context(result)
        messages = build_message(context, query, image_url, chat_history)

    elif intent == "shop_information":
        # general information search
        general_information = embed.embedded_retrieve_general_information(cur, conn, query)
        messages = build_message(general_information, query, image_url, chat_history)

    elif image_url:
        pass
    else:
        # pass straight to the generator
        messages = build_message(context, query, image_url, chat_history)

    # Generate answer using LLM
    answer = generator.generate_answer(messages, session_id, user_id)

    if guard.guard_check__response(answer):
        return jsonify({'message': 'Hệ thống gặp trục trặc, vui lòng thử lại sau!'}), 500

    # Save chat history to database only if user_id is provided
    if user_id:
        try:
            save_chat_history(cur, conn, user_id, session_id, query, answer)
        except Exception as e:
            print(f"Warning: Failed to save chat history: {str(e)}")
            # Continue even if saving fails - don't break the API response
    mysql.close_connection()
    return jsonify({'message': answer}), 200


@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    """
    Get chat history for a user
    ---
    tags:
      - Chat History
    summary: Get chat history by user_id
    description: |
      This endpoint retrieves the chat history for a specific user based on their user_id.
      Returns all chat conversations associated with that user, ordered by most recent first.

    parameters:
      - name: user_id
        in: query
        type: string
        required: true
        description: |
          The user identifier to retrieve chat history for.
          This should match the user_id used when making queries.
        example: "user_12345"

    responses:
      200:
        description: Successful retrieval of chat history
        schema:
          type: object
          properties:
            user_id:
              type: string
              description: The user identifier
            chat_history:
              type: array
              description: List of chat conversations
              items:
                type: object
                properties:
                  user_session_id:
                    type: string
                    description: Session identifier for this conversation
                  user_chat:
                    type: string
                    description: The user's query/chat message
                  response:
                    type: string
                    description: The system's response
                  created_at:
                    type: string
                    format: date-time
                    description: Timestamp when the conversation occurred
        examples:
          application/json:
            user_id: "user_12345"
            chat_history:
              - user_session_id: "session_abc123"
                user_chat: "tôi muốn mua hoa màu vàng"
                response: "Chúng tôi có nhiều loại hoa màu vàng..."
                created_at: "2025-12-08T10:30:00"
      
      400:
        description: Bad request - Missing user_id parameter
        schema:
          type: object
          properties:
            message:
              type: string
        examples:
          application/json:
            message: "Missing required parameter 'user_id'"
      
      404:
        description: No chat history found for the user
        schema:
          type: object
          properties:
            message:
              type: string
            user_id:
              type: string
        examples:
          application/json:
            message: "No chat history found for this user"
            user_id: "12345"
      
      500:
        description: Internal server error
        schema:
          type: object
          properties:
            message:
              type: string
        examples:
          application/json:
            message: "Error retrieving chat history"

    produces:
      - application/json
    """
    user_id = request.args.get('user_id')
    cur, conn = connect()
    if not user_id:
        return jsonify({'message': "Missing required parameter 'user_id'"}), 400
    
    try:
        chat_history = load_chat_history(cur, conn, user_id)
        
        if not chat_history:
            return jsonify({
                'message': 'No chat history found for this user',
                'user_id': user_id,
                'chat_history': []
            }), 404
        
        return jsonify({
            'user_id': user_id,
            'chat_history': chat_history
        }), 200
        
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return jsonify({'message': 'Error retrieving chat history'}), 500


@app.route('/chat_history', methods=['PUT'])
def reset_chat_history():
    """
        Reset chat history session
        ---
        tags:
          - Chat History
        summary: Reset chat history session
        description: Creates a new session ID and resets the current chat session for the user
        parameters:
          - in: query
            name: reset_session
            required: true
            schema:
              type: boolean
              enum:
                - true
                - false
            description: Flag to indicate whether to reset the session
          - in: query
            name: user_id
            required: true
            schema:
              type: string
            description: ID of the user
        responses:
          200:
            description: Session reset successfully
          400:
            description: Bad request
          500:
            description: Internal server error
    """
    try:
        data = request.get_json()
        reset_session = data.get("reset_session") or None
        user_id = data.get("user_id")

        if not reset_session:
            return jsonify({"error": "reset_session parameter is required"}), 400

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        # Generate new session ID
        session_id = int(datetime.now().strftime("%Y%m%d%H%M%S"))

        # Reset session
        reset_session_method(cur, conn, session_id, user_id)

        return jsonify({
            "message": "Session reset successfully",
            "new_session_id": session_id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommendations/batch', methods=['POST'])
def get_batch_recommendations():
    """
    Get recommendations for multiple users at once
    ---
    tags:
      - Recommendations
    summary: Get batch recommendations for multiple users
    description: |
      Retrieves personalized product recommendations for multiple users in a single request.
      Useful for generating recommendations for user lists or dashboards.
      Maximum 100 users per request.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - user_ids
            properties:
              user_ids:
                type: array
                items:
                  type: integer
                description: List of user IDs to get recommendations for
                minItems: 1
                maxItems: 100
                example: [123, 456, 789]
              limit:
                type: integer
                description: Number of recommendations per user
                default: 5
                minimum: 1
                maximum: 50
                example: 5
          examples:
            basic_request:
              summary: Basic batch request
              value:
                user_ids: [123, 456, 789]
                limit: 5
            custom_limit:
              summary: Request with custom limit
              value:
                user_ids: [100, 200, 300, 400]
                limit: 10
    responses:
      200:
        description: Successfully retrieved batch recommendations
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  example: true
                results:
                  type: object
                  description: Map of user_id to their recommendations
                  additionalProperties:
                    oneOf:
                      - type: array
                        items:
                          type: object
                          properties:
                            product_id:
                              type: integer
                              example: 456
                            name:
                              type: string
                              example: "Hoa Hồng Đỏ"
                            price:
                              type: number
                              format: float
                              example: 500000
                            images:
                              type: string
                              example: "https://example.com/image.jpg"
                            score:
                              type: number
                              format: float
                              example: 0.89
                            reason:
                              type: string
                              example: "Based on your interests"
                      - type: object
                        properties:
                          error:
                            type: string
                            example: "User not found"
            examples:
              success_response:
                summary: Successful batch response
                value:
                  success: true
                  results:
                    "123":
                      - product_id: 456
                        name: "Hoa Hồng Đỏ"
                        price: 500000
                        images: "https://example.com/rose.jpg"
                        score: 0.89
                        reason: "Based on your recent purchases"
                      - product_id: 789
                        name: "Hoa Tulip Vàng"
                        price: 350000
                        images: "https://example.com/tulip.jpg"
                        score: 0.85
                        reason: "Popular in your area"
                    "456":
                      - product_id: 101
                        name: "Hoa Cẩm Chướng"
                        price: 200000
                        images: "https://example.com/carnation.jpg"
                        score: 0.92
                        reason: "Matches your preferences"
                    "789":
                      error: "User not found"
              partial_success:
                summary: Partial success with some errors
                value:
                  success: true
                  results:
                    "123":
                      - product_id: 456
                        name: "Hoa Hồng Đỏ"
                        price: 500000
                        images: "https://example.com/rose.jpg"
                        score: 0.89
                        reason: "Based on your interests"
                    "456":
                      error: "Insufficient user data"
      400:
        description: Bad request - Invalid input
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  example: false
                error:
                  type: string
                  example: "Invalid user_ids (max 100)"
            examples:
              empty_user_ids:
                summary: Empty user IDs list
                value:
                  success: false
                  error: "Invalid user_ids (max 100)"
              too_many_users:
                summary: Too many user IDs
                value:
                  success: false
                  error: "Invalid user_ids (max 100)"
      500:
        description: Internal server error
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  example: false
                error:
                  type: string
                  example: "Database connection failed"
            examples:
              error_response:
                summary: Server error
                value:
                  success: false
                  error: "Unable to process batch recommendations"
    """
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        limit = data.get('limit', 5)

        if not user_ids or len(user_ids) > 100:
            return jsonify({
                'success': False,
                'error': 'Invalid user_ids (max 100)'
            }), 400

        mysql = MysqlService()
        conn = mysql.return_connection()

        recommender = get_recommender(conn)
        results = {}

        for user_id in user_ids:
            try:
                recs = recommender.get_recommendations(user_id, limit)
                results[str(user_id)] = recs
            except Exception as e:
                results[str(user_id)] = {'error': str(e)}
        mysql.close_connection()
        return jsonify({
            'success': True,
            'results': results
        }), 200

    except Exception as e:
        print(f"Error in get_batch_recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/recommendations/personalized/<int:user_id>', methods=['GET'])
def get_personalized_recommendations(user_id: int):
    """
    Get personalized product recommendations for a user
    ---
    tags:
      - Recommendations
    summary: Get personalized product recommendations
    description: |
      Retrieves personalized product recommendations for a specific user based on their
      browsing history, preferences, and interaction patterns.
    parameters:
      - name: user_id
        in: path
        description: Unique identifier of the user
        required: true
        schema:
          type: integer
          example: 123
      - name: limit
        in: query
        description: Number of recommendations to return
        required: false
        schema:
          type: integer
          default: 10
          minimum: 1
          maximum: 50
          example: 10
    responses:
      200:
        description: Successfully retrieved recommendations
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  example: true
                user_id:
                  type: integer
                  example: 123
                recommendations:
                  type: array
                  items:
                    type: object
                    properties:
                      product_id:
                        type: integer
                        example: 456
                      name:
                        type: string
                        example: "Hoa Hồng Đỏ"
                      price:
                        type: number
                        format: float
                        example: 500000
                      images:
                        type: string
                        example: "https://example.com/image.jpg"
                      score:
                        type: number
                        format: float
                        minimum: 0
                        maximum: 1
                        example: 0.89
                        description: Recommendation confidence score
                      reason:
                        type: string
                        example: "Based on your interests"
                        description: Explanation for why this product is recommended
                count:
                  type: integer
                  example: 10
                  description: Total number of recommendations returned
            examples:
              success_response:
                summary: Successful recommendation response
                value:
                  success: true
                  user_id: 123
                  recommendations:
                    - product_id: 456
                      name: "Hoa Hồng Đỏ"
                      price: 500000
                      images: "https://example.com/rose.jpg"
                      score: 0.89
                      reason: "Based on your recent purchases"
                    - product_id: 789
                      name: "Hoa Tulip Vàng"
                      price: 350000
                      images: "https://example.com/tulip.jpg"
                      score: 0.85
                      reason: "Popular in your area"
                  count: 2
      500:
        description: Internal server error
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  example: false
                error:
                  type: string
                  example: "Database connection failed"
            examples:
              error_response:
                summary: Error response
                value:
                  success: false
                  error: "Unable to generate recommendations"
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', default=10, type=int)
        limit = min(max(limit, 1), 50)  # Limit between 1 and 50

        mysql = MysqlService()
        conn = mysql.return_connection()
        # Get recommendations
        recommender = get_recommender(conn)
        recommendations = recommender.get_recommendations(user_id, limit)

        # Add reason text
        for rec in recommendations:
            rec['reason'] = _generate_reason(rec)

        mysql.close_connection()
        return jsonify({
            'success': True,
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        }), 200

    except Exception as e:
        print(f"Error in get_personalized_recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
