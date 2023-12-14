import os
import json
import logging
import time
import yaml

import spotipy
from langchain.requests import Requests
from langchain.llms import OpenAI

from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT

logger = logging.getLogger()


def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ['SPOTIPY_CLIENT_ID'] = config['spotipy_client_id']
    os.environ['SPOTIPY_CLIENT_SECRET'] = config['spotipy_client_secret']
    os.environ['SPOTIPY_REDIRECT_URI'] = config['spotipy_redirect_uri']

    query_idx = 1

    log_dir = os.path.join("logs", "restgpt_spotify")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint()), logging.FileHandler(os.path.join(log_dir, f"{query_idx}.log"), mode='w', encoding='utf-8')],
    )
    logger.setLevel(logging.INFO)

    with open("specs/spotify_oas.json") as f:
        raw_api_spec = json.load(f)

    api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)

    scopes = list(raw_api_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
    access_token = spotipy.util.prompt_for_user_token(scope=','.join(scopes))
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    requests_wrapper = Requests(headers=headers)

    llm = OpenAI(model_name="text-davinci-003", temperature=0.0, max_tokens=700)
    # llm = OpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.0)
    rest_gpt = RestGPT(llm, api_spec=api_spec, scenario='spotify', requests_wrapper=requests_wrapper, simple_parser=False)

    queries = json.load(open('datasets/spotify.json', 'r'))
    queries = [item['query'] for item in queries]

    query = queries[query_idx - 1]
    
    logger.info(f"Query: {query}")

    start_time = time.time()
    rest_gpt.run(query)
    logger.info(f"Execution Time: {time.time() - start_time}")

if __name__ == '__main__':
    main()
