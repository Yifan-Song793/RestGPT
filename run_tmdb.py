import os
import json
import logging
import time
import yaml

from langchain.requests import Requests
from langchain.llms import OpenAI

from utils import reduce_openapi_spec, ColorPrint, MyRotatingFileHandler
from model import RestGPT

logger = logging.getLogger()


def run(query, api_spec, requests_wrapper, simple_parser=False):
    llm = OpenAI(model_name="text-davinci-003", temperature=0.0, max_tokens=256)
    # llm = OpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.0, max_tokens=256)
    rest_gpt = RestGPT(llm, api_spec=api_spec, scenario='tmdb', requests_wrapper=requests_wrapper, simple_parser=simple_parser)

    logger.info(f"Query: {query}")

    start_time = time.time()
    rest_gpt.run(query)
    logger.info(f"Execution Time: {int(time.time() - start_time)} seconds")


def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ["TMDB_ACCESS_TOKEN"] = config['tmdb_access_token']

    log_dir = os.path.join("logs", "restgpt_tmdb")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    file_handler = MyRotatingFileHandler(os.path.join(log_dir, f"tmdb.log"), encoding='utf-8')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint()), file_handler],
    )
    logger.setLevel(logging.INFO)

    with open("specs/tmdb_oas.json") as f:
        raw_tmdb_api_spec = json.load(f)

    api_spec = reduce_openapi_spec(raw_tmdb_api_spec, only_required=False)

    access_token = os.environ["TMDB_ACCESS_TOKEN"]
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    requests_wrapper = Requests(headers=headers)

    queries = json.load(open('datasets/tmdb.json', 'r'))
    queries = [item['query'] for item in queries]

    for idx, query in enumerate(queries, 1):
        try:
            print('#' * 20 + f" Query-{idx} " + '#' * 20)
            run(query, api_spec, requests_wrapper, simple_parser=False)
        except Exception as e:
            print(f"Query: {query}\nError: {e}")
        finally:
            file_handler.doRollover()
    

if __name__ == '__main__':
    main()
