import os
import json
import logging
import time
import yaml

import spotipy
from langchain.requests import Requests
from langchain import OpenAI

from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT

logger = logging.getLogger()

import streamlit as st
st.title("RestGPT")
st.subheader("An LLM-based autonomous agent controlling real-world applications via RESTful APIs")

def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

    # Define a function to get values from st.secrets or fallback to config.yaml
    def get_secret_or_config(key):
        if key in st.secrets:
            return st.secrets[key]
        elif key in config and config[key] != '':
            return config[key]
        else:
            raise ValueError(f"Key '{key}' not found in secrets or config")

    # Set environment variables
    os.environ["OPENAI_API_KEY"] = get_secret_or_config('openai_api_key')
    os.environ["TMDB_ACCESS_TOKEN"] = get_secret_or_config('tmdb_access_token')
    os.environ['SPOTIPY_CLIENT_ID'] = get_secret_or_config('spotipy_client_id')
    os.environ['SPOTIPY_CLIENT_SECRET'] = get_secret_or_config('spotipy_client_secret')
    os.environ['SPOTIPY_REDIRECT_URI'] = get_secret_or_config('spotipy_redirect_uri')
        
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint())],
    )
    logger.setLevel(logging.INFO)

    scenario = st.selectbox(
        'Which API you want to play with',
        ('TMDB', 'Spotify'))

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

    llm = OpenAI(model_name="text-davinci-003", temperature=0.0, max_tokens=-1)
    rest_gpt = RestGPT(llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=False)

    if scenario == 'tmdb':
        query_example = 'What is the movie directed by SS Rajamouli in 2022' # Since ChatGPT is trained with data till sep 2021 we ask query for 2022
    elif scenario == 'spotify':
        query_example = "Add Summertime Sadness by Lana Del Rey in my first playlist"
    st.write(f"Example instruction: {query_example}")
    query = st.text_input('Query', query_example)
    if st.button("Run", type="primary"):
        logger.info(f"Query: {query}")
        start_time = time.time()
        output = rest_gpt.run(query)
        st.write(output)
        logger.info(f"Execution Time: {time.time() - start_time}")

if __name__ == '__main__':
    main()
