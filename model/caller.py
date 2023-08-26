import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
import yaml
import time
import re
import requests

import tiktoken

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.requests import RequestsWrapper
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM

from utils import simplify_json, get_matched_endpoint, ReducedOpenAPISpec, fix_json_error
from .parser import ResponseParser, SimpleResponseParser


logger = logging.getLogger(__name__)




CALLER_PROMPT = """You are an agent that gets a sequence of API calls and given their documentation, should execute them and return the final response.
If you cannot complete them and run into issues, you should explain the issue. If you're able to resolve an API call, you can retry the API call. When interacting with API objects, you should extract ids for inputs to other API calls but ids and names for outputs returned to the User.
Your task is to complete the corresponding api calls according to the plan.


Here is documentation on the API:
Base url: {api_url}
Endpoints:
{api_docs}

If the API path contains "{{}}", it means that it is a variable and you should replace it with the appropriate value. For example, if the path is "/users/{{user_id}}/tweets", you should replace "{{user_id}}" with the user id. "{{" and "}}" cannot appear in the url.

You can use http request method, i.e., GET, POST, DELETE, PATCH, PUT, and generate the corresponding parameters according to the API documentation and the plan.
The input should be a JSON string which has 3 base keys: url, description, output_instructions
The value of "url" should be a string.
The value of "description" should describe what the API response is about. The description should be specific.
The value of "output_instructions" should be instructions on what information to extract from the response, for example the id(s) for a resource(s) that the POST request creates. Note "output_instructions" MUST be natural language and as verbose as possible! It cannot be "return the full response". Output instructions should faithfully contain the contents of the api calling plan and be as specific as possible. The output instructions can also contain conditions such as filtering, sorting, etc.
If you are using GET method, add "params" key, and the value of "params" should be a dict of key-value pairs.
If you are using POST, PATCH or PUT methods, add "data" key, and the value of "data" should be a dict of key-value pairs.
Remember to add a comma after every value except the last one, ensuring that the overall structure of the JSON remains valid.

Example 1:
Operation: POST
Input: {{
    "url": "https://api.twitter.com/2/tweets",
    "params": {{
        "tweet.fields": "created_at"
    }}
    "data": {{
        "text": "Hello world!"
    }},
    "description": "The API response is a twitter object.",
    "output_instructions": "What is the id of the new twitter?"
}}

Example 2:
Operation: GET
Input: {{
    "url": "https://api.themoviedb.org/3/person/5026/movie_credits",
    "description": "The API response is the movie credit list of Akira Kurosawa (id 5026)",
    "output_instructions": "What are the names and ids of the movies directed by this person?"
}}

Example 3:
Operation: PUT
Input: {{
    "url": "https://api.spotify.com/v1/me/player/volume",
    "params": {{
        "volume_percent": "20"
    }},
    "description": "Set the volume for the current playback device."
}}

I will give you the background information and the plan you should execute.
Background: background information which you can use to execute the plan, e.g., the id of a person.
Plan: the plan of API calls to execute

You should execute the plan faithfully and give the Final Answer as soon as you successfully call the planned APIs, don't get clever and make up steps that don't exist in the plan. Do not make up APIs that don't exist in the plan. For example, if the plan is "GET /search/person to search for the director "Lee Chang dong"", do not call "GET /person/{{person_id}}/movie_credits" to get the credit of the person.

Starting below, you must follow this format:

Background: background information which you can use to execute the plan, e.g., the id of a person.
Plan: the plan of API calls to execute
Thought: you should always think about what to do
Operation: the request method to take, should be one of the following: GET, POST, DELETE, PATCH, PUT
Input: the input to the operation
Response: the output of the operation
Thought: I am finished executing the plan (or, I cannot finish executing the plan without knowing some other information.)
Execution Result: based on the API response, the execution result of the API calling plan.

The execution result should satisfy the following conditions:
1. The execution result must contain "Execution Result:" prompt.
2. You should reorganize the response into natural language based on the plan. For example, if the plan is "GET /search/person to search for the director "Lee Chang dong"", the execution result should be "Successfully call GET /search/person to search for the director "Lee Chang dong". The id of Lee Chang dong is xxxx". Do not use pronouns if possible. For example, do not use "The id of this person is xxxx".
3. If the plan includes expressions such as "most", you should choose the first item from the response. For example, if the plan is "GET /trending/tv/day to get the most trending TV show today", you should choose the first item from the response.
4. The execution result should be natural language and as verbose as possible. It must contain the information needed in the plan.

Begin!

Background: {background}
Plan: {api_plan}
Thought: {agent_scratchpad}
"""



class Caller(Chain):
    llm: BaseLLM
    api_spec: ReducedOpenAPISpec
    scenario: str
    requests_wrapper: RequestsWrapper
    max_iterations: Optional[int] = 15
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "force"
    simple_parser: bool = False
    with_response: bool = False
    output_key: str = "result"

    def __init__(self, llm: BaseLLM, api_spec: ReducedOpenAPISpec, scenario: str, requests_wrapper: RequestsWrapper, simple_parser: bool = False, with_response: bool = False) -> None:
        super().__init__(llm=llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=simple_parser, with_response=with_response)

    @property
    def _chain_type(self) -> str:
        return "RestGPT Caller"

    @property
    def input_keys(self) -> List[str]:
        return ["api_plan"]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
            self.max_execution_time is not None
            and time_elapsed >= self.max_execution_time
        ):
            return False

        return True
    
    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Response: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought: "
    
    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]
    
    def _construct_scratchpad(
        self, history: List[Tuple[str, str]]
    ) -> str:
        if len(history) == 0:
            return ""
        scratchpad = ""
        for i, (plan, execution_res) in enumerate(history):
            scratchpad += self.llm_prefix.format(i + 1) + plan + "\n"
            scratchpad += self.observation_prefix + execution_res + "\n"
        return scratchpad

    def _get_action_and_input(self, llm_output: str) -> Tuple[str, str]:
        if "Execution Result:" in llm_output:
            return "Execution Result", llm_output.split("Execution Result:")[-1].strip()
        # \s matches against tab/newline/whitespace
        regex = r"Operation:[\s]*(.*?)[\n]*Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # TODO: not match, just return
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        if action not in ["GET", "POST", "DELETE", "PUT"]:
            raise NotImplementedError
        
        # avoid error in the JSON format
        action_input = fix_json_error(action_input)

        return action, action_input
    
    def _get_response(self, action: str, action_input: str) -> str:
        action_input = action_input.strip().strip('`')
        left_bracket = action_input.find('{')
        right_bracket = action_input.rfind('}')
        action_input = action_input[left_bracket:right_bracket + 1]
        try:
            data = json.loads(action_input)
        except json.JSONDecodeError as e:
            raise e
        
        desc = data.get("description", "No description")
        query = data.get("output_instructions", None)

        params, request_body = None, None
        if action == "GET":
            if 'params' in data:
                params = data.get("params")
                response = self.requests_wrapper.get(data.get("url"), params=params)
            else:
                response = self.requests_wrapper.get(data.get("url"))
        elif action == "POST":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.post(data["url"], params=params, data=request_body)
        elif action == "PUT":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.put(data["url"], params=params, data=request_body)
        elif action == "DELETE":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.delete(data["url"], params=params, json=request_body)
        else:
            raise NotImplementedError
        
        if isinstance(response, requests.models.Response):
            if response.status_code != 200:
                return response.text
            response_text = response.text
        elif isinstance(response, str):
            response_text = response
        else:
            raise NotImplementedError
        
        return response_text, params, request_body, desc, query
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        intermediate_steps: List[Tuple[str, str]] = []

        api_plan = inputs['api_plan']
        api_url = self.api_spec.servers[0]['url']
        matched_endpoints = get_matched_endpoint(self.api_spec, api_plan)
        endpoint_docs_by_name = {name: docs for name, _, docs in self.api_spec.endpoints}
        api_doc_for_caller = ""
        assert len(matched_endpoints) == 1, f"Found {len(matched_endpoints)} matched endpoints, but expected 1."
        endpoint_name = matched_endpoints[0]
        tmp_docs = deepcopy(endpoint_docs_by_name.get(endpoint_name))
        if 'responses' in tmp_docs and 'content' in tmp_docs['responses']:
            if 'application/json' in tmp_docs['responses']['content']:
                tmp_docs['responses'] = tmp_docs['responses']['content']['application/json']['schema']['properties']
            elif 'application/json; charset=utf-8' in tmp_docs['responses']['content']:
                tmp_docs['responses'] = tmp_docs['responses']['content']['application/json; charset=utf-8']['schema']['properties']
        if not self.with_response and 'responses' in tmp_docs:
            tmp_docs.pop("responses")
        tmp_docs = yaml.dump(tmp_docs)
        encoder = tiktoken.encoding_for_model('text-davinci-003')
        encoded_docs = encoder.encode(tmp_docs)
        if len(encoded_docs) > 1500:
            tmp_docs = encoder.decode(encoded_docs[:1500])
        api_doc_for_caller += f"== Docs for {endpoint_name} == \n{tmp_docs}\n"

        caller_prompt = PromptTemplate(
            template=CALLER_PROMPT,
            partial_variables={
                "api_url": api_url,
                "api_docs": api_doc_for_caller,
            },
            input_variables=["api_plan", "background", "agent_scratchpad"],
        )
        
        caller_chain = LLMChain(llm=self.llm, prompt=caller_prompt)

        while self._should_continue(iterations, time_elapsed):
            scratchpad = self._construct_scratchpad(intermediate_steps)
            caller_chain_output = caller_chain.run(api_plan=api_plan, background=inputs['background'], agent_scratchpad=scratchpad, stop=self._stop)
            logger.info(f"Caller: {caller_chain_output}")

            action, action_input = self._get_action_and_input(caller_chain_output)
            if action == "Execution Result":
                return {"result": action_input}
            response, params, request_body, desc, query = self._get_response(action, action_input)

            called_endpoint_name = action + ' ' + json.loads(action_input)['url'].replace(api_url, '')
            called_endpoint_name = get_matched_endpoint(self.api_spec, called_endpoint_name)[0]
            api_path = api_url + called_endpoint_name.split(' ')[-1]
            api_doc_for_parser = endpoint_docs_by_name.get(called_endpoint_name)
            if self.scenario == 'spotify' and endpoint_name == "GET /search":
                if params is not None and 'type' in params:
                    search_type = params['type'] + 's'
                else:
                    params_in_url = json.loads(action_input)['url'].split('&')
                    for param in params_in_url:
                        if 'type=' in param:
                            search_type = param.split('=')[-1] + 's'
                            break
                api_doc_for_parser['responses']['content']['application/json']["schema"]['properties'] = {search_type: api_doc_for_parser['responses']['content']['application/json']["schema"]['properties'][search_type]}

            if not self.simple_parser:
                response_parser = ResponseParser(
                    llm=self.llm,
                    api_path=api_path,
                    api_doc=api_doc_for_parser,
                )
            else:
                response_parser = SimpleResponseParser(
                    llm=self.llm,
                    api_path=api_path,
                    api_doc=api_doc_for_parser,
                )

            params_or_data = {
                "params": params if params is not None else "No parameters",
                "data": request_body if request_body is not None else "No request body",
            }
            parsing_res = response_parser.run(query=query, response_description=desc, api_param=params_or_data, json=response)
            logger.info(f"Parser: {parsing_res}")

            intermediate_steps.append((caller_chain_output, parsing_res))

            iterations += 1
            time_elapsed = time.time() - start_time

        return {"result": caller_chain_output}
