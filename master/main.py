from flask import Flask, request
from flask_restful import Api, Resource
from flasgger import Swagger

app = Flask(__name__)
api = Api(app)
swagger = Swagger(app)