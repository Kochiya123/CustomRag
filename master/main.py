from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_restful import Api
from flasgger import Swagger
import os
import psycopg2
import tiktoken

from langfuse import Langfuse

from master.db_module.connect import connect
from master.generate_text_module.generator import Generator_llm, build_context, build_message
from master.embed_module.embed_llm import Embed_llm


from master.guardrail.guardrail import Guardrail
from master.query_classify_module.query_classify import extract_info

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

def save_chat_history(cur, conn, user_id, user_session_id, user_chat, response):
    """Save chat history to database"""
    try:
        user_session_id_str = str(user_session_id) if user_session_id is not None else None
        
        # If user_id exists, remove it from insert parameters (don't include user_id column)
        if user_id:
            # Insert without user_id column
            cur.execute("""
                INSERT INTO chat_history (user_id, user_session_id, user_chat, response)
                VALUES (%s, %s, %s, %s)
            """, (user_id, user_session_id_str, user_chat, response))
        
        conn.commit()
        print("Chat history saved successfully")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error saving chat history: {error}")
        try:
            conn.rollback()
        except Exception as rollback_error:
            print(f"Rollback error: {rollback_error}")

def load_chat_history(cur, conn, user_id):
    """Load chat history from database based on user_id"""
    try:
        user_id_str = str(user_id) if user_id is not None else None
        cur.execute("""
            SELECT user_session_id, user_chat, response, created_at
            FROM chat_history
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id_str,))
        results = cur.fetchall()
        
        # Format results as list of dictionaries
        chat_history = []
        for row in results:
            chat_history.append({
                'user_session_id': row[0],
                'user_chat': row[1],
                'response': row[2],
                'created_at': row[3].isoformat() if row[3] else None
            })
        
        return chat_history
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error loading chat history: {error}")
        try:
            conn.rollback()
        except Exception as rollback_error:
            print(f"Rollback error: {rollback_error}")
        return []

def count_tokens(text, model="gpt-4o"):
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback: rough estimation (1 token ≈ 4 characters for English, but Vietnamese might be different)
        return len(text) // 3

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
    
    result = embed.embedded_add_single_column_product(cur, conn, product_string, product_name, product_id, category_id, price)
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
    
    result = embed.embedded_update_single_column_product(cur, conn, product_id, product_string, product_name, category_id, price)
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
        example:
          query: "tôi muốn mua hoa màu vàng cho ngày của mẹ"
          user_id: "user_12345"
          session_id: "session_abc123"
          image_url: "https://cdn.pixabay.com/photo/2023/03/14/11/19/flower-7852094_1280.jpg"

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
    session_id = data.get("session_id") or None
    image_url = data.get("image_url") or None


    # Get optional tracking parameters
    #user_id = request.args.get('user_id') or None
    #session_id = request.args.get('session_id') or None
    #image_url = request.args.get('image_url') or None

    # Get query and optional image_url from query parameters
    #query = request.args.get('query')
    user_chat = query

    context = []
    
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

    info = extract_info(cur,query)
    context = ""
    
    # Route all queries through embedding-based retrieval
    if info['category']:
        # Category-based query - use embedding-based retrieval with category filtering
        context = embed.embedded_retrieve_category_information(cur, conn, query, info['category'])
        if not context:
            return jsonify({'message': f'Chúng tôi không tìm thấy sản phẩm nào cho danh mục "{info["category"]}"!'}), 404
        # Format reranked results
        if isinstance(context, list) and context:
            formatted_products = []
            for product in context[:5]:  # Limit to top 10
                if len(product) >= 2:
                    product_id = product[0]
                    product_text = product[1]
                    # Fetch product_name and price from database using product_id
                    cur.execute(
                        "SELECT product_name, price FROM product_vector WHERE product_id = %s",
                        (product_id,)
                    )
                    result = cur.fetchone()
                    if result:
                        product_name, price = result
                        formatted_products.append((product_id, product_name, product_text, price))
            if formatted_products:
                context = build_context(formatted_products)
            else:
                context = ""
        messages = build_message(context, query)
    elif info['flower']:
            # Get text-based retrieval results
            text_results = embed.embedded_retrieve_products_information(cur,conn, query, info['preference'], info['flower'], info['price'])
            
            # If image_url is provided, compute image-text similarity and combine
            if image_url:
                try:
                    # Compute image-to-text similarity
                    image_results = embed.compute_image_text_similarity(cur, conn, image_url)
                    
                    # Combine text and image results
                    # text_results format: (product_id, product_text, score) or (product_id, product_text)
                    # image_results format: (product_id, product_text, similarity_score)
                    
                    # Convert text_results to consistent format with scores
                    text_results_with_scores = []
                    for item in text_results if isinstance(text_results, list) else []:
                        if len(item) >= 2:
                            product_id = item[0]
                            product_text = item[1]
                            # If score exists, use it; otherwise set to 0.5
                            score = item[2] if len(item) >= 3 else 0.5
                            text_results_with_scores.append((product_id, product_text, score))
                    
                    # Combine both results: merge by product_id and average scores
                    combined_results = {}
                    
                    # Add text results
                    for product_id, product_text, score in text_results_with_scores:
                        combined_results[product_id] = {
                            'id': product_id,
                            'product_text': product_text,
                            'text_score': score,
                            'image_score': 0.0,
                            'combined_score': score
                        }
                    
                    # Add/update with image results
                    for product_id, product_text, image_score in image_results:
                        if product_id in combined_results:
                            # Average the scores
                            combined_results[product_id]['image_score'] = image_score
                            combined_results[product_id]['combined_score'] = (
                                combined_results[product_id]['text_score'] + image_score
                            ) / 2.0
                        else:
                            combined_results[product_id] = {
                                'id': product_id,
                                'product_text': product_text,
                                'text_score': 0.0,
                                'image_score': image_score,
                                'combined_score': image_score
                            }
                    
                    # Convert to list and sort by combined_score
                    combined_list = list(combined_results.values())
                    combined_list.sort(key=lambda x: x['combined_score'], reverse=True)
                    
                    # Rerank the combined results
                    # Prepare data for reranker: list of (product_id, product_text)
                    rerank_data = [(item['id'], item['product_text']) for item in combined_list]
                    
                    if rerank_data:
                        reranked_results = reranker.rerank_query(query, rerank_data)
                        # reranked_results format: (product_id, product_text, score)
                        context = reranked_results
                    else:
                        context = text_results
                except Exception as e:
                    print(f"Error processing image: {e}")
                    # Fallback to text-only results
                    context = text_results
            else:
                # No image, use text results only
                context = text_results
            
            # Format reranked results if needed
            # Context format: (product_id, product_text, score) or (product_id, product_text)
            if isinstance(context, list) and context:
                formatted_products = []
                for product in context[:5]:  # Limit to top 10
                    # Handle both formats: (product_id, product_text, score) or (product_id, product_text)
                    if len(product) >= 2:
                        product_id = product[0]  # product_id is always first element
                        product_text = product[1]  # product_text is always second element
                        # Fetch product_name and price from database using product_id
                        cur.execute(
                            "SELECT product_name, price FROM product_vector WHERE product_id = %s",
                            (product_id,)
                        )
                        result = cur.fetchone()
                        if result:
                            product_name, price = result
                            # Format: (product_id, product_name, product_text, price)
                            formatted_products.append((product_id, product_name, product_text, price))
                if formatted_products:
                    context = build_context(formatted_products)
                else:
                    context = ""
            messages = build_message(context, query)
    else:
        # General information - use embedding-based retrieval
        general_info = embed.embedded_retrieve_general_information(cur, conn, query)
        if general_info:
            # general_info is a tuple from database, extract the text field
            # Assuming structure: (general_id, general_text, general_embedding)
            context = general_info[1] if len(general_info) > 1 else str(general_info)
        else:
            context = ""
        messages = build_message(context, query)


    # Generate answer using LLM
    answer = generator.generate_answer(messages, session_id, user_id)

    if guard.guard_check__response(answer):
        return jsonify({'message': 'Hệ thống gặp trục trặc, vui lòng thử lại sau!'}), 500

    # Save chat history to database only if user_id is provided
    if user_id:
        try:
            save_chat_history(cur, conn, user_id, session_id, user_chat, answer)
        except Exception as e:
            print(f"Warning: Failed to save chat history: {str(e)}")
            # Continue even if saving fails - don't break the API response

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
