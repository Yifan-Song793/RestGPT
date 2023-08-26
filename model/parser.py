import json
import logging
from typing import Any, Dict, List, Optional
import sys
from io import StringIO

from pydantic import BaseModel, Field, Extra

import tiktoken

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM

from utils import simplify_json

logger = logging.getLogger(__name__)

RESPONSE_SCHEMA_MAX_LENGTH = 5000


CODE_PARSING_SCHEMA_TEMPLATE = """Here is an API response schema from an OAS and a query. 
The API's response will follow the schema and be a JSON. 
Assume you are given a JSON response which is stored in a python dict variable called 'data', your task is to generate Python code to extract information I need from the API response.
Note: I will give you 'data', do not make up one, just reference it in your code.
Please print the final result as brief as possible. If the result is a list, just print it in one sentence. Do not print each item in a new line.
The example result format are:
"The release date of the album is 2002-11-03"
"The id of the person is 12345"
"The movies directed by Wong Kar-Wai are In the Mood for Love (843), My Blueberry Nights (1989), Chungking Express (11104)"
Note you should generate only Python code.
DO NOT use fields that are not in the response schema.

API: {api_path}
API description: {api_description}
Parameters or body for this API call:
{api_param}

Response JSON schema defined in the OAS:
{response_schema}

Example:
{response_example}

The response is about: {response_description}

Query: {query}

The code you generate should satisfy the following requirements:
1. The code you generate should contain the filter in the query. For example, if the query is "what is the name and id of the director of this movie" and the response is the cast and crew for the movie, instead of directly selecting the first result in the crew list (director_name = data['crew'][0]['name']), the code you generate should have a filter for crews where the job is a "Director" (item['job'] == 'Director').
2. If the response is something about X, e.g., the movies credits of Lee Chang-dong, then the filter condition cannot include searching for X (e.g., Lee Chang-dong). For example, the API response is the movie credits of Akira Kurosawa and the instruction is what are the ids of the movies directed by him. Then the your code should not contain "movie['title'] == 'Akira Kurosawa'" or "movie['name'] == 'Akira Kurosawa'"
3. Do not use f-string in the print function. Use "format" instead. For example, use "print('The release date of the album is {{}}'.format(date))" instead of "print(f'The release date of the album is {{date}}')
4. Please print the final result as brief as possible. If the result is a list, just print it in one sentence. Do not print each item in a new line.

Begin!
Python Code:
"""

CODE_PARSING_RESPONSE_TEMPLATE = """Here is an API response JSON snippet with its corresponding schema and a query. 
The API's response JSON follows the schema.
Assume the JSON response is stored in a python dict variable called 'data', your task is to generate Python code to extract information I need from the API response.
Please print the final result.
The example result format are:
"The release date of the album is 2002-11-03"
"The id of the person is 12345"
Note you should generate only Python code.
DO NOT use fields that are not in the response schema.

API: {api_path}
API description: {api_description}
Parameters for this API call:
{api_param}

Response JSON schema defined in the OAS:
{response_schema}

JSON snippet:
{json}

Query: {query}
Python Code:
"""

LLM_PARSING_TEMPLATE = """Here is an API JSON response with its corresponding API description:

API: {api_path}
API description: {api_description}
Parameters for this API call:
{api_param}

JSON response:
{json}

The response is about: {response_description}

====
Your task is to extract some information according to these instructions: {query}
When working with API objects, you should usually use ids over names.
If the response indicates an error, you should instead output a summary of the error.

Output:
"""

LLM_SUMMARIZE_TEMPLATE = """Here is an API JSON response with its corresponding API description:

API: {api_path}
API description: {api_description}
Parameters for this API call:
{api_param}

JSON response:
{json}

The response is about: {response_description}

====
Your task is to extract some information according to these instructions: {query}
If the response does not contain the needed information, you should translate the response JSON into natural language.
If the response indicates an error, you should instead output a summary of the error.

Output:
"""

CODE_PARSING_EXAMPLE_TEMPLATE = """Here is an API response schema and a query. 
The API's response will follow the schema and be a JSON. 
Assume you are given a JSON response which is stored in a python dict variable called 'data', your task is to generate Python code to extract information I need from the API response.
Please print the final result.
The example result format are:
Note you should generate only Python code.
DO NOT use fields that are not in the response schema.

API: {api_path}
API description: {api_description}

Response example:
{response_example}

Query: {query}
Python Code:
"""


POSTPROCESS_TEMPLATE = """Given a string, due to the maximum context length, the final item/sentence may be truncated and incomplete. First, remove the final truncated incomplete item/sentence. Then if the list are in brackets "[]", add bracket in the tail to make it a grammarly correct list. You should just output the final result.

Example:
Input: The ids and names of the albums from Lana Del Rey are [{{'id': '5HOHne1wzItQlIYmLXLYfZ', 'name': "Did you know that there's a tunnel under Ocean Blvd"}}, {{'id': '2wwCc6fcyhp1tfY3J6Javr', 'name': 'Blue Banisters'}}, {{'id': '6Qeos
Output: The ids and names of the albums from Lana Del Rey are [{{'id': '5HOHne1wzItQlIYmLXLYfZ', 'name': "Did you know that there's a tunnel under Ocean Blvd"}}, {{'id': '2wwCc6fcyhp1tfY3J6Javr', 'name': 'Blue Banisters'}}]

Begin!
Input: {truncated_str}
Output: 
"""


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")

    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, self.globals, self.locals)
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            print(str(e))
            output = None
        return output


class ResponseParser(Chain):
    """Implements Program-Aided Language Models."""

    llm: BaseLLM
    code_parsing_schema_prompt: PromptTemplate = None
    code_parsing_response_prompt: PromptTemplate = None
    llm_parsing_prompt: PromptTemplate = None
    postprocess_prompt: PromptTemplate = None
    python_globals: Optional[Dict[str, Any]] = None
    python_locals: Optional[Dict[str, Any]] = None
    encoder: tiktoken.Encoding = None
    max_json_length_1: int = 500
    max_json_length_2: int = 2000
    max_output_length: int = 500
    output_key: str = "result"
    return_intermediate_steps: bool = False


    def __init__(self, llm: BaseLLM, api_path: str, api_doc: Dict, with_example: bool = False) -> None:
        if 'responses' not in api_doc or 'content' not in api_doc['responses']:
            llm_parsing_prompt = PromptTemplate(
                template=LLM_SUMMARIZE_TEMPLATE,
                partial_variables={
                    "api_path": api_path,
                    "api_description": api_doc['description'],
                },
                input_variables=["query", "json", "api_param", "response_description"]
            )
            super().__init__(llm=llm, llm_parsing_prompt=llm_parsing_prompt)
            return

        if 'application/json' in api_doc['responses']['content']:
            response_schema = json.dumps(api_doc['responses']['content']['application/json']["schema"]['properties'], indent=4)
        elif 'application/json; charset=utf-8' in api_doc['responses']['content']:
            response_schema = json.dumps(api_doc['responses']['content']['application/json; charset=utf-8']["schema"]['properties'], indent=4)
        encoder = tiktoken.encoding_for_model('text-davinci-003')
        encoded_schema = encoder.encode(response_schema)
        max_schema_length = 2500
        if len(encoded_schema) > max_schema_length:
            response_schema = encoder.decode(encoded_schema[:max_schema_length]) + '...'
        # if len(response_schema) > RESPONSE_SCHEMA_MAX_LENGTH:
        #     response_schema = response_schema[:RESPONSE_SCHEMA_MAX_LENGTH] + '...'
        if with_example and 'examples' in api_doc['responses']['content']['application/json']:
            response_example = simplify_json(api_doc['responses']['content']['application/json']["examples"]['response']['value'])
            response_example = json.dumps(response_example, indent=4)
        else:
            response_example = "No example provided"
        code_parsing_schema_prompt = PromptTemplate(
            template=CODE_PARSING_SCHEMA_TEMPLATE,
            partial_variables={
                "api_path": api_path,
                "api_description": api_doc['description'],
                "response_schema": response_schema,
                "response_example": response_example,
            },
            input_variables=["query", "response_description", "api_param"]
        )
        code_parsing_response_prompt = PromptTemplate(
            template=CODE_PARSING_RESPONSE_TEMPLATE,
            partial_variables={
                "api_path": api_path,
                "api_description": api_doc['description'],
                "response_schema": response_schema,
            },
            input_variables=["query", "json", "api_param"]
        )
        llm_parsing_prompt = PromptTemplate(
            template=LLM_PARSING_TEMPLATE,
            partial_variables={
                "api_path": api_path,
                "api_description": api_doc['description'],
            },
            input_variables=["query", "json", "api_param", "response_description"]
        )
        postprocess_prompt = PromptTemplate(
            template=POSTPROCESS_TEMPLATE,
            input_variables=["truncated_str"]
        )

        super().__init__(llm=llm, 
                         code_parsing_schema_prompt=code_parsing_schema_prompt, 
                         code_parsing_response_prompt=code_parsing_response_prompt, 
                         llm_parsing_prompt=llm_parsing_prompt, 
                         postprocess_prompt=postprocess_prompt, 
                         encoder=encoder)

    @property
    def _chain_type(self) -> str:
        return "RestGPT Parser"

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return ["query", "json", "api_param", "response_description"]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, "intermediate_steps"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if self.code_parsing_schema_prompt is None or inputs['query'] is None:
            extract_code_chain = LLMChain(llm=self.llm, prompt=self.llm_parsing_prompt)
            output = extract_code_chain.predict(query=inputs['query'], json=inputs['json'], api_param=inputs['api_param'], response_description=inputs['response_description'])
            return {"result": output}
        
        extract_code_chain = LLMChain(llm=self.llm, prompt=self.code_parsing_schema_prompt)
        code = extract_code_chain.predict(query=inputs['query'], response_description=inputs['response_description'], api_param=inputs['api_param'])
        logger.info(f"Code: \n{code}")
        json_data = json.loads(inputs["json"])
        repl = PythonREPL(_globals={"data": json_data})
        res = repl.run(code)
        output = res

        if output is None or len(output) == 0:
            extract_code_chain = LLMChain(llm=self.llm, prompt=self.code_parsing_response_prompt)
            json_data = json.loads(inputs["json"])
            encoded_json = self.encoder.encode(inputs["json"])
            if len(encoded_json) > self.max_json_length_1:
                simplified_json_data = self.encoder.decode(encoded_json[:self.max_json_length_1]) + '...'
            else:
                simplified_json_data = inputs["json"]
            # simplified_json_data = json.dumps(simplify_json(json_data), indent=4)
            code = extract_code_chain.predict(query=inputs['query'], json=simplified_json_data, api_param=inputs['api_param'])
            logger.info(f"Code: \n{code}")
            repl = PythonREPL(_globals={"data": json_data})
            res = repl.run(code)
            output = res

        if output is None or len(output) == 0:
            extract_code_chain = LLMChain(llm=self.llm, prompt=self.llm_parsing_prompt)
            if len(encoded_json) > self.max_json_length_2:
                simplified_json_data = self.encoder.decode(encoded_json[:self.max_json_length_2]) + '...'
            output = extract_code_chain.predict(query=inputs['query'], json=simplified_json_data, api_param=inputs['api_param'], response_description=inputs['response_description'])

        encoded_output = self.encoder.encode(output)
        if len(encoded_output) > self.max_output_length:
            output = self.encoder.decode(encoded_output[:self.max_output_length])
            logger.info(f"Output too long, truncating to {self.max_output_length} tokens")
            postprocess_chain = LLMChain(llm=self.llm, prompt=self.postprocess_prompt)
            output = postprocess_chain.predict(truncated_str=output)

        return {"result": output}

    


class SimpleResponseParser(Chain):
    """Implements Program-Aided Language Models."""

    llm: BaseLLM
    llm_parsing_prompt: PromptTemplate = None
    encoder: tiktoken.Encoding = None
    max_json_length: int = 1000
    output_key: str = "result"
    return_intermediate_steps: bool = False


    def __init__(self, llm: BaseLLM, api_path: str, api_doc: Dict, with_example: bool = False) -> None:
        if 'responses' not in api_doc or 'content' not in api_doc['responses']:
            llm_parsing_prompt = PromptTemplate(
                template=LLM_SUMMARIZE_TEMPLATE,
                partial_variables={
                    "api_path": api_path,
                    "api_description": api_doc['description'],
                },
                input_variables=["query", "json", "api_param", "response_description"]
            )
            encoder = tiktoken.encoding_for_model('text-davinci-003')
            super().__init__(llm=llm, llm_parsing_prompt=llm_parsing_prompt, encoder=encoder)
            return

        llm_parsing_prompt = PromptTemplate(
            template=LLM_PARSING_TEMPLATE,
            partial_variables={
                "api_path": api_path,
                "api_description": api_doc['description'],
            },
            input_variables=["query", "json", "api_param", "response_description"]
        )

        encoder = tiktoken.encoding_for_model('text-davinci-003')

        super().__init__(llm=llm, llm_parsing_prompt=llm_parsing_prompt, encoder=encoder)

    @property
    def _chain_type(self) -> str:
        return "RestGPT Parser"

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return ["query", "json", "api_param", "response_description"]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, "intermediate_steps"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if inputs['query'] is None:
            extract_code_chain = LLMChain(llm=self.llm, prompt=self.llm_parsing_prompt)
            output = extract_code_chain.predict(query=inputs['query'], json=inputs['json'], api_param=inputs['api_param'], response_description=inputs['response_description'])
            return {"result": output}
        
        encoded_json = self.encoder.encode(inputs["json"])
        extract_code_chain = LLMChain(llm=self.llm, prompt=self.llm_parsing_prompt)
        if len(encoded_json) > self.max_json_length:
            encoded_json = self.encoder.decode(encoded_json[:self.max_json_length]) + '...'
        output = extract_code_chain.predict(query=inputs['query'], json=encoded_json, api_param=inputs['api_param'], response_description=inputs['response_description'])

        return {"result": output}

    