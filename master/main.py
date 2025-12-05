from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flasgger import Swagger
from huggingface_hub import login
import os

from langfuse import Langfuse

from master.connect import connect
from master.generator import Generator_llm, build_context, build_message
from master.embed_llm import Embed_llm


from master.guardrail import Guardrail

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
            - context
            - image_url
          properties:
            context:
              type: string
              description: The combine string of product to embed
              example: "Hoa màu vàng 123456"
            image_url:
              type: string
              description: The image url of product to embed
              example: "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg"
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
              example: "Missing required fields: context, image_url"
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
    
    context = data.get('context')
    image_url = data.get('image_url')
    
    if not context or not image_url:
        return jsonify({'message': 'Missing required fields: context, image_url'}), 400
    
    result = embed.embedded_add_single_column(cur, conn, context, image_url)
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
            - context
            - image_url
          properties:
            context:
              type: string
              description: The combine string of product to embed
              example: "Hoa màu vàng 123456"
            image_url:
              type: string
              description: The url image string of product to embed
              example: "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg"
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
    
    context = data.get('context')
    image_url = data.get('image_url')
    
    if not context or not image_url:
        return jsonify({'message': 'Missing required fields: context, image_url'}), 400
    
    result = embed.embedded_update_single_column(cur, conn, product_id, context, image_url)
    if result:
        conn.commit()  # Commit the transaction
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
    description: This endpoint delete a single product by its ID
    parameters:
      - name: product_id
        in: path
        type: string
        required: true
        description: The ID of the product to embed
        example: "123"
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

    result = embed.embedded_delete_single_column(cur, conn, product_id)
    if result == product_id:
        return jsonify({'message': f'Delete product {product_id} successfully'}), 200
    return jsonify({'message': f'Delete single product {product_id} failed: '}), 500

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
    # Get query from query parameter
    query = request.args.get('query')
    
    if not query:
        return jsonify({'message': 'Missing query parameter "query"'}), 400
    
    # Get optional tracking parameters
    user_id = request.args.get('user_id') or None  # Fallback to IP if not provided
    session_id = request.args.get('session_id') or None
    
    # Content moderation check on input
    #if guard.guard_check__response(query):
        #return jsonify({'message': 'Câu hỏi chứa nội dung không phù hợp'}), 400

    product_context = embed.retrieval_vector_new(cur, conn, query)
        
    if product_context is None:
        return jsonify({'message': 'Chúng tôi không thể tìm thấy sản phẩm nào phù hợp với mô tả của bạn!'}), 404

    messages = build_message(product_context, query)

    # Generate answer using LLM
    answer = generator.generate_answer(messages, session_id, user_id)

    #if guard.guard_check__response(answer):
        #return jsonify({'message': 'Hệ thống gặp trục trặc, vui lòng thử lại sau!'}), 500

    return jsonify({'message': answer}), 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
