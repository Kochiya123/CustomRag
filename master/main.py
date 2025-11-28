import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flasgger import Swagger
from flask_swagger import swagger as swagger_auto
from flask_swagger_ui import get_swaggerui_blueprint
from huggingface_hub import login

from master.connect import connect
from master.generator import (
    Generator_llm,
    build_context,
    build_messages,
    generate_reponse,
)
from master.embed_llm import (
    Embed_llm,
    embedded_all_column,
    embedded_single_column,
    embedded_update_multiple_column,
    read_column,
    retrieval_vector,
)

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

app = Flask(__name__)
swagger = Swagger(app)

generator = Generator_llm()
embed = Embed_llm()

SWAGGER_URL = "/swagger-ui"
API_SPEC_URL = "/swagger.json"
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_SPEC_URL,
    config={"app_name": "My API"},
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


def _connect():
    return connect()


@app.route("/product", methods=["POST"])
def embed_all_vector_null():
    """
    Embed all products whose vector is NULL
    ---
    responses:
      200:
        description: Embed all products successfully
      500:
        description: Embed all products failed
    """
    cur, conn = _connect()
    read_column(cur)
    result = embedded_all_column(cur, conn, embed)
    if result == 1:
        return jsonify({"message": "Embed all products successfully"}), 200
    return jsonify({"message": f"Embed all products failed: {result}"}), 500


@app.route("/product/<product_id>", methods=["PUT"])
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
    cur, conn = _connect()
    result = embedded_single_column(cur, conn, embed, product_id)
    if result == product_id:
        return jsonify({"message": f"Embed {product_id} product successfully"}), 200
    return jsonify({"message": f"Embed single product failed: {result}"}), 500


@app.route("/product", methods=["PUT"])
def embed_all_update():
    """
    Update embeddings for multiple products
    ---
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            ids:
              type: array
              items:
                type: integer
    responses:
      200:
        description: Embed all products successfully
      400:
        description: Missing ids payload
      500:
        description: Embed all products failed
    """
    payload = request.get_json(silent=True) or {}
    ids = payload.get("ids")
    if not ids:
        return (
            jsonify({"message": "Embed all products failed: ids payload is required"}),
            400,
        )
    cur, conn = _connect()
    result = embedded_update_multiple_column(cur, conn, embed, ids)
    if result == ids:
        return jsonify({"message": "Embed all products successfully"}), 200
    return jsonify({"message": f"Embed all products failed: {result}"}), 500


@app.route("/<query>", methods=["GET"])
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
      404:
        description: Product not found
      500:
        description: System error
    """
    products = retrieval_vector(query, embed)
    if not products:
        return jsonify({"message": "Sản phẩm không có trong cửa hàng"}), 404

    context = build_context(products)
    messages = build_messages(context, query)
    answer = generate_reponse(generator, messages)

    return jsonify({"message": answer}), 200


@app.route("/swagger.json")
def get_swagger():
    """
    Generate Swagger specification
    ---
    responses:
      200:
        description: Swagger specification generated
    """
    swag = swagger_auto(app)
    swag["info"]["version"] = "1.0"
    swag["info"]["title"] = "My API"
    return jsonify(swag)