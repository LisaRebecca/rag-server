from pydantic import BaseModel, Field
from helpers.exception import CustomException
from helpers.logger import logging
import requests
import sys
import http.client

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
        
    def pipe(self, body: dict, __user__: dict):
        print(f"pipe:{__name__}")
        logging.info("Entered Pipe...")
        print("Entered Pipe...")
        
        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        logging.info(f"Headers: {headers}")

        model_id = body["model"][body["model"].find(".") + 1:]
        logging.info(f"Model_id: {model_id}")

        payload = {**body, "model": model_id}
        logging.info(f"payload: {payload}")

        URL = f"{self.valves.OPENAI_API_BASE_URL}/v1/chat/completions"
        logging.info(f"URL: {URL}")

        # Enable HTTP connection debugging
        http.client.HTTPConnection.debuglevel = 1
        logging.basicConfig(level=logging.DEBUG)

        try:
            r = requests.post(
                url=URL,
                json=payload,
                headers=headers,
                stream=True,
            )
            logging.info({r.status_code, r.text})
            print(r.status_code, r.text)  # See if the response comes back properly
            r.raise_for_status()  # Raise an exception for HTTP errors

            if body.get("stream", False):
                return r.iter_lines()
            else:
                logging.info(f"Pipe response :) : {r.json()}")
                print(f"Pipe response: {r.json()}")
                return r.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}", exc_info=True)
            raise CustomException(e, sys)

        except Exception as e:
            logging.error(f"Unexpected error during POST request: {e}", exc_info=True)
            raise CustomException(e, sys)