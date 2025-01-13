from pydantic import BaseModel, Field
from helpers.exception import CustomException
from helpers.logger import logging
import requests
import sys

class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="OPENAI/",
            description="Prefix to be added before model names.",
        )
        OPENAI_API_BASE_URL: str = Field(
            default="http://localhost:8090",
            description="Base URL for accessing OpenAI API endpoints.",
        )
        OPENAI_API_KEY: str = Field(
            default="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",
            description="API key for authenticating requests to the OpenAI API.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        if self.valves.OPENAI_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }

                r = requests.get(
                    f"{self.valves.OPENAI_API_BASE_URL}/models", headers=headers
                )
                models = r.json()
                return [
                    {
                        "id": model["id"],
                        "name": f'{self.valves.NAME_PREFIX}{model.get("name", model["id"])}',
                    }
                    for model in models["data"]
                    if "gpt" in model["id"]
                ]
            except Exception as e:
                return [
                    {
                        "id": "error",
                        "name": "Error fetching models. Please check your API Key.",
                    },
                ]
                raise CustomException(e, sys)
        else:
            return [
                {
                    "id": "error",
                    "name": "API Key not provided.",
                },
            ]
        
    def pipe(self, body: dict, __user__:dict):
        print(f"pipe:{__name__}")
        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        # Extract model id from the model name
        model_id = body["model"][body["model"].find(".") + 1 :]

        # Update the model id in the body
        payload = {**body, "model": model_id}
        try:
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )
            r.raise_for_status()

            if body.get("stream", False):
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            raise CustomException(e, sys)