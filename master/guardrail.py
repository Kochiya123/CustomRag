import os

from flask.cli import load_dotenv
from openai import OpenAI

class Guardrail:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPEN_AI_TOKEN")
        if not api_key:
            raise ValueError("OPEN_AI_TOKEN environment variable is not set")
        self.client = OpenAI(api_key=api_key)

    def guard_check__response(self, query):
        """
        Check if the input query contains inappropriate content.
        
        Args:
            query (str): The text to check for moderation
            
        Returns:
            int: 1 if content is flagged, 0 if safe
        """
        try:
            response = self.client.moderations.create(
                model="omni-moderation-latest",
                input=query
            )
            if response.results and len(response.results) > 0:
                if response.results[0].flagged:
                    return 1
            return 0
        except Exception as e:
            print(f"Guardrail check failed: {str(e)}")
            return 0



