from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flasgger import Swagger
from huggingface_hub import login
import os

from langfuse import Langfuse

from master.connect import connect
from master.generator import Generator_llm, build_context, build_message, build_message_general
from master.embed_llm import Embed_llm


from master.guardrail import Guardrail
from master.query_classify import extract_info

from master.rerank_llm import Rerank

load_dotenv()

#create langfuse for logging and tracing llm call
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    # optional if self-hosted
    # base_url="http://localhost:8000"
)

login(token=os.getenv("HF_TOKEN"))
cur, conn = connect()
generator = Generator_llm()
embed = Embed_llm()
guard = Guardrail()
reranker = Rerank()

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

@app.route('/delivery', methods=['POST'])
def embed_delivery():
    """
    Add delivery information with embedding
    ---
    tags:
      - Delivery
    summary: Add delivery information and generate its embedding
    description: |
      This endpoint adds delivery information (shipping policies, delivery times, etc.) 
      and generates a vector embedding for semantic search.
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - delivery_text
          properties:
            delivery_text:
              type: string
              description: Delivery information text to embed
              example: "Giao hàng miễn phí trong nội thành, thời gian giao hàng 2-3 ngày"
    responses:
      200:
        description: Delivery information added successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Add delivery information successfully"
      400:
        description: Missing required fields
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Missing required fields: delivery_text"
      500:
        description: Failed to add delivery information
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Add delivery information failed"
    """
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Request body must be JSON'}), 400

    delivery_text = data.get('delivery_text')
    if not delivery_text:
        return jsonify({'message': 'Missing required fields: delivery_text'}), 400

    result = embed.embedded_add_delivery_information(cur, conn, delivery_text)
    if result:
        return jsonify({'message': f'Add delivery information successfully'}), 200
    return jsonify({'message': f'Add delivery information failed'}), 500

@app.route('/delivery/<delivery_id>', methods=['PUT'])
def embed_update_delivery_information(delivery_id):
    """
    Update delivery information and its embedding
    ---
    tags:
      - Delivery
    summary: Update existing delivery information and regenerate its embedding
    description: |
      This endpoint updates delivery information and regenerates its vector embedding.
    parameters:
      - name: delivery_id
        in: path
        type: integer
        required: true
        description: The unique identifier of the delivery information to update
        example: 1
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - delivery_text
          properties:
            delivery_text:
              type: string
              description: Updated delivery information text
              example: "Giao hàng miễn phí toàn quốc, thời gian giao hàng 1-2 ngày"
    responses:
      200:
        description: Delivery information updated successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Update delivery information successfully"
      400:
        description: Missing required fields
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Missing required fields: delivery_text"
      500:
        description: Failed to update delivery information
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Update delivery information failed"
    """
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Request body must be JSON'}), 400
    delivery_text = data.get('delivery_text')
    if not delivery_text:
        return jsonify({'message': 'Missing required fields: delivery_text'}), 400
    result = embed.embedded_update_delivery_information(cur, conn, delivery_id, delivery_text)
    if result:
        return jsonify({'message': f'Update delivery information successfully'}), 200
    return jsonify({'message': f'Update delivery information failed'}), 500

@app.route('/delivery/<delivery_id>', methods=['DELETE'])
def embed_delete_delivery_information(delivery_id):
    """
    Delete delivery information
    ---
    tags:
      - Delivery
    summary: Delete delivery information and its embedding
    description: |
      This endpoint deletes delivery information and removes its vector embedding from the database.
    parameters:
      - name: delivery_id
        in: path
        type: integer
        required: true
        description: The unique identifier of the delivery information to delete
        example: 1
    responses:
      200:
        description: Delivery information deleted successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Delete delivery information successfully"
      500:
        description: Failed to delete delivery information
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Delete delivery information failed"
    """
    result = embed.embedded_delete_delivery_information(cur, conn, delivery_id)
    if result:
        return jsonify({'message': f'Delete delivery information successfully'}), 200
    return jsonify({'message': f'Delete delivery information failed'}), 500



@app.route('/', methods=['GET'])
def get_answer():
    """
    Generate answer for a query
    ---
    tags:
      - Query
    summary: Get answer for a query
    description: |
      This endpoint generates an answer for a given Vietnamese query using RAG.
      The query must be provided via URL parameter `query`.

    parameters:
      - name: query
        in: query
        type: string
        required: true
        description: |
          The user query in Vietnamese. This is the question or request that the user wants answered.
          The system will search for relevant products and generate an answer based on the context.
        example: "tôi muốn mua hoa màu vàng cho ngày của mẹ"
      
      - name: user_id
        in: query
        type: string
        required: false
        description: |
          Optional user identifier for tracking and analytics purposes.
          Used for Langfuse tracing to associate queries with specific users.
        example: "user_12345"
      
      - name: session_id
        in: query
        type: string
        required: false
        description: |
          Optional session identifier for tracking conversation sessions.
          Used for Langfuse tracing to group related queries in a session.
        example: "session_abc123"

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
        description: Bad request - Missing query parameter or inappropriate content detected
        schema:
          type: object
          properties:
            message:
              type: string
              description: Error message explaining what went wrong
        examples:
          application/json:

            - message: "Câu hỏi chứa nội dung không phù hợp"
      
      404:
        description: Product not found
        schema:
          type: object
          properties:
            message:
              type: string
              description: Message indicating no relevant products were found
        examples:
          application/json:
            message: "Chúng tôi không thể tìm thấy sản phẩm nào phù hợp với mô tả của bạn!"
      
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
    # Get optional tracking parameters
    user_id = request.args.get('user_id') or None
    session_id = request.args.get('session_id') or None

    # Get query from query parameter
    query = request.args.get('query')

    context = []
    
    if not query:
        return jsonify({'message': 'Missing query parameter "query"'}), 400

    # Content moderation check on input
    # if guard.guard_check__response(query):
    # return jsonify({'message': 'Câu hỏi chứa nội dung không phù hợp'}), 400

    info = extract_info(cur,query)
    context = []
    
    if all(value is None for value in info.values()):
        messages = build_message_general(query)
    else:
        if info['flower']:
            context = embed.embedded_retrieve_products_information(cur,conn, query, info['preference'], info['flower'], info['price'])
        elif info['delivery']:
            context = embed.embedded_retrieve_delivery_information(cur,conn,query)
        
        messages = build_message(context, query)

    # Generate answer using LLM
    answer = generator.generate_answer(messages, session_id, user_id)

    #if guard.guard_check__response(answer):
        #return jsonify({'message': 'Hệ thống gặp trục trặc, vui lòng thử lại sau!'}), 500

    return jsonify({'message': answer}), 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
