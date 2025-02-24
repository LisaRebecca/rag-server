import json
import sys
import logging
import aiohttp
import time

class CustomException(Exception):
    pass

# Load configuration
with open('fastapi_RAG_container/config.json') as config_file:
    config = json.load(config_file)

# Function to get model configuration
def get_model_config(model_name):
    logging.info(f"Model name: {model_name}")
    return config['llm'].get(model_name)

async def query_university_endpoint(query, model_name='techxgenus'):
    start_time = time.time()
    model_config = get_model_config(model_name)
    if not model_config:
        raise CustomException(f"Model configuration for '{model_name}' not found")

    api_url = model_config['base_url']
    logging.info(f"API URL: {api_url}")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {model_config['api_key']}"
    }
    data = {
        "model": model_config['model'],
        "messages": [{"role": "user", "content": query}]
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(api_url, headers=headers, json=data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        answer = response_data['choices'][0]['message']['content']
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Generation took {elapsed_time:.4f} seconds")
                    else:
                        answer = 'No answer found in the response'
                    return answer
                else:
                    return f"Error: {response.status} - {await response.text()}"
        except Exception as e:
            raise CustomException(e, sys)