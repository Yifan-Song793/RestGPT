import os
import json
import logging
import datetime
import time
import yaml

import spotipy
from langchain.requests import Requests
from langchain import OpenAI

from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT

logger = logging.getLogger()


def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ["TMDB_ACCESS_TOKEN"] = config['tmdb_access_token']
    os.environ['SPOTIPY_CLIENT_ID'] = config['spotipy_client_id']
    os.environ['SPOTIPY_CLIENT_SECRET'] = config['spotipy_client_secret']
    os.environ['SPOTIPY_REDIRECT_URI'] = config['spotipy_redirect_uri']
        
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint())],
    )
    logger.setLevel(logging.INFO)

    scenario = input("Please select a scenario (TMDB/Spotify): ")
    scenario = scenario.lower()

    if scenario == 'tmdb':
        with open("specs/tmdb_oas.json") as f:
            raw_tmdb_api_spec = json.load(f)

        api_spec = reduce_openapi_spec(raw_tmdb_api_spec, only_required=False)

        access_token = os.environ["TMDB_ACCESS_TOKEN"]
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
    elif scenario == 'spotify':
        with open("specs/spotify_oas.json") as f:
            raw_api_spec = json.load(f)

        api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)

        scopes = list(raw_api_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
        access_token = spotipy.util.prompt_for_user_token(scope=','.join(scopes))
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    requests_wrapper = Requests(headers=headers)

    llm = OpenAI(model_name="text-davinci-003", temperature=0.0, max_tokens=700)
    rest_gpt = RestGPT(llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=False)

    if scenario == 'tmdb':
        query_example = "Give me the number of movies directed by Sofia Coppola"
    elif scenario == 'spotify':
        query_example = "Add Summertime Sadness by Lana Del Rey in my first playlist"
    print(f"Example instruction: {query_example}")
    query = input("Please input an instruction (Press ENTER to use the example instruction): ")
    if query == '':
        query = query_example
    
    logger.info(f"Query: {query}")

    start_time = time.time()
    rest_gpt.run(query)
    logger.info(f"Execution Time: {time.time() - start_time}")

if __name__ == '__main__':
    main()
