from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flasgger import Swagger
from huggingface_hub import login
import os

from master.connect import connect
from master.generator import Generator_llm, build_context, build_message
from master.embed_llm import Embed_llm


from master.guardrail import Guardrail

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
cur, conn = connect()
generator = Generator_llm()
embed = Embed_llm()
guard = Guardrail()

app = Flask(__name__)
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

"""
@app.route('/product', methods=['POST'])
def embed_all_vector_null():
    
    Embed all products
    ---
    tags:
      - Products
    summary: Embed all products with null vectors
    description: This endpoint embeds all products that have null vector values
    responses:
      200:
        description: Embed all products successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: Embed all products successfully
      500:
        description: Embed all products failed
        schema:
          type: object
          properties:
            message:
              type: string
              example: Embed all products failed
    
    result = embed.embedded_all_column_vector_null(cur, conn)
    if result == 1:
        return jsonify({'message': 'Embed all products successfully'}), 200
    return jsonify({'message': 'Embed all products failed: ' + str(result)}), 500
"""


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
              example: Missing required fields: context, image_url
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
              example: Missing required fields: context, image_url
      500:
        description: Update the embed failed
        schema:
          type: object
          properties:
            message:
              type: string
              example: Update embed 123 product failed
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
              example: Missing required fields: product_id
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


"""
@app.route('/product', methods=['PUT'])
def embed_all_update():
    
    Embed all products (update)
    ---
    tags:
      - Products
    summary: Update embeddings for all products
    description: This endpoint updates embeddings for all products
    responses:
      200:
        description: Embed all products successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: Embed all products successfully
      500:
        description: Embed all products failed
        schema:
          type: object
          properties:
            message:
              type: string
              example: Embed all products failed
    
    result = embed.embedded_update_all_column(cur, conn)
    if result == 1:
        return jsonify({'message': 'Embed all products successfully'}), 200
    return jsonify({'message': 'Embed all products failed: ' + str(result)}), 500
"""

@app.route('/<query>', methods=['GET'])
def get_answer(query):
    """
    Generate answer for a query
    ---
    tags:
      - Query
    summary: Get answer for a query
    description: This endpoint generates an answer for a given query using RAG
    parameters:
      - name: query
        in: path
        type: string
        required: true
        description: Query to generate an answer
        example: "What products do you have?"
    responses:
      200:
        description: Successful answer
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Here are the products available..."
      400:
        description: Inappropriate content
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Câu hỏi chứa nội dung không phù hợp"
      404:
        description: Product not found
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Sản phẩm không có trong cửa hàng"
      500:
        description: System error
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Hệ thống gặp trục trặc, vui lòng thử lại sau!"
    """
    #if guard.guard_check__response(query):
        #return jsonify({'message': 'Câu hỏi chứa nội dung không phù hợp'}), 400

    product_context = embed.retrieval_vector_new(cur, conn, query)
    if product_context is None:
        return jsonify({'message': 'Chúng tôi không thể tìm thấy sản phẩm nào phù hợp với mô tả của bạn!'}), 404


    messages = build_message(product_context, query)

    answer = generator.generate_answer(messages)

    #if guard.guard_check__response(answer):
        #return jsonify({'message': 'Hệ thống gặp trục trặc, vui lòng thử lại sau!'}), 500

    return jsonify({'message': answer}), 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
