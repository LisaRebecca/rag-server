import httpx
import sys
import requests
from helpers.logger import logging
from helpers.exception import CustomException

api_url = 'http://lme49.cs.fau.de:5000/v1/chat/completions'

headers = {
    'Content-Type': 'application/json',

    'Authorization': 'Bearer xFhGltj52Gn'
}

def query_university_endpoint(query): # Provided by Sebastian - Requires Cisco FAU VPN
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
        raise CustomException(e, sys)