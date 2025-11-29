from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flasgger import Swagger, swag_from
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
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

@app.route('/product', methods=['POST'])
def embed_all_vector_null():
    """
    Embed all products
    ---
    responses:
      200:
        description: Embed all products successfully
      500:
        description: Embed all products failed
    """
    result = embed.embedded_all_column_vector_null(cur, conn)
    if result == 1:
        return jsonify({'message': 'Embed all products successfully'}), 200
    return jsonify({'message': 'Embed all products failed: ' + str(result)}), 500


@app.route('/product/<product_id>', methods=['PUT'])
def embed_single(product_id):
    """
    Embed single product
    ---
    parameters:
      - name: product_id
        in: path
        type: string
        required: true
        description: The ID of the product to embed
    responses:
      200:
        description: Embed single product successfully
      500:
        description: Embed single product failed
    """
    result = embed.embedded_single_column(cur, conn, product_id)
    if result == product_id:
        return jsonify({'message': f'Embed {product_id} product successfully'}), 200
    return jsonify({'message': 'Embed single product failed: ' + str(result)}), 500


@app.route('/product', methods=['PUT'])
def embed_all_update():
    """
    Embed all products (update)
    ---
    responses:
      200:
        description: Embed all products successfully
      500:
        description: Embed all products failed
    """
    result = embed.embedded_update_all_column(cur, conn)
    if result == 1:
        return jsonify({'message': 'Embed all products successfully'}), 200
    return jsonify({'message': 'Embed all products failed: ' + str(result)}), 500


@app.route('/<query>', methods=['GET'])
def get_answer(query):
    """
    Generate answer for a query
    ---
    parameters:
      - name: query
        in: path
        type: string
        required: true
        description: Query to generate an answer
    responses:
      200:
        description: Successful answer
      400:
        description: Inappropriate content
      404:
        description: Product not found
      500:
        description: System error
    """
    #if guard.guard_check__response(query):
        #return jsonify({'message': 'Câu hỏi chứa nội dung không phù hợp'}), 400

    product = embed.retrieval_vector(cur, conn, query)
    if product is None:
        return jsonify({'message': 'Sản phẩm không có trong cửa hàng'}), 404

    context = build_context(product)
    messages = build_message(context, query)
    answer = generator.generate_answer(messages)

    #if guard.guard_check__response(answer):
        #return jsonify({'message': 'Hệ thống gặp trục trặc, vui lòng thử lại sau!'}), 500

    return jsonify({'message': answer}), 200

@app.route('/swagger')
def get_swagger():
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "My API"
    return jsonify(swag)

SWAGGER_URL = '/swagger-ui'
API_URL = '/swagger'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "My API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
