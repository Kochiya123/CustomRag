import os

from openai import OpenAI
import json

class Guardrail:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))


    def guard_check__response(self, query):
        response = self.client.moderations.create(
         model="omni-moderation-latest",
         input=query
        )
        result = json.loads(response.content)
        if result["results"][0]["flagged"] == "true":
            return 1
        return 0



