import json
import requests
import sys
import logging
import aiohttp

class CustomException(Exception):
    pass

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Function to get model configuration
def get_model_config(model_name):
    logging.info(f"Model name: {model_name}")
    return config['llm'].get(model_name)

async def query_university_endpoint(query, model_name='techxgenus'):
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
                    else:
                        answer = 'No answer found in the response'
                    return answer
                else:
                    return f"Error: {response.status} - {await response.text()}"
        except Exception as e:
            raise CustomException(e, sys)

    """try:
        response = requests.post(api_url, headers=headers, json=data, verify=False)
        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                answer = response_data['choices'][0]['message']['content']
            else:
                answer = 'No answer found in the response'
            return answer
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        raise CustomException(e, sys)"""
    

"""import httpx
import sys
import requests
from helpers.logger import logging
from helpers.exception import CustomException

api_url = 'http://lme49.cs.fau.de:5000/v1/chat/completions'

headers = {
    'Content-Type': 'application/json',

    'Authorization': 'Bearer xFhGltj52Gn'
}

async def query_university_endpoint(query): # Provided by Sebastian - Requires Cisco FAU VPN
    data = {
    "model": "TechxGenus_Mistral-Large-Instruct-2407-AWQ",

    "messages": [{"role": "user", "content": query}]
    }

    try:
        response = requests.post(api_url, headers=headers, json=data, verify=False)

        if response.status_code == 200:
            response_data = response.json()

            if 'choices' in response_data and len(response_data['choices']) > 0:
                answer = response_data['choices'][0]['message']['content']

            else:
                answer = 'No answer found in the response'

            return answer
        
        else:
            return f"Error: {response.status_code} - {response.text}"
        
    except Exception as e:
        raise CustomException(e, sys)"""
