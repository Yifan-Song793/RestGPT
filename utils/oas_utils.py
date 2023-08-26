"""Quick and dirty representation for OpenAPI specs."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union


def dereference_refs(spec_obj: dict, full_spec: dict) -> Union[dict, list]:
    """Try to substitute $refs.

    The goal is to get the complete docs for each endpoint in context for now.

    In the few OpenAPI specs I studied, $refs referenced models
    (or in OpenAPI terms, components) and could be nested. This code most
    likely misses lots of cases.
    """

    def _retrieve_ref_path(path: str, full_spec: dict) -> dict:
        components = path.split("/")
        if components[0] != "#":
            raise RuntimeError(
                "All $refs I've seen so far are uri fragments (start with hash)."
            )
        out = full_spec
        for component in components[1:]:
            out = out[component]
        return out

    def _dereference_refs(
        obj: Union[dict, list], stop: bool = False
    ) -> Union[dict, list]:
        if stop:
            return obj
        obj_out: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "$ref":
                    # stop=True => don't dereference recursively.
                    return _dereference_refs(
                        _retrieve_ref_path(v, full_spec), stop=False
                    )
                elif isinstance(v, list):
                    obj_out[k] = [_dereference_refs(el) for el in v]
                elif isinstance(v, dict):
                    obj_out[k] = _dereference_refs(v)
                else:
                    obj_out[k] = v
            return obj_out
        elif isinstance(obj, list):
            return [_dereference_refs(el) for el in obj]
        else:
            return obj

    return _dereference_refs(spec_obj)


def merge_allof_properties(obj):
    def merge(to_merge):
        merged = {'properties': {}, 'required': [], 'type': 'object'}
        for partial_schema in to_merge:
            if 'allOf' in partial_schema:
                tmp = merge(partial_schema['allOf'])
                merged['properties'].update(tmp['properties'])
                if 'required' in tmp:
                    merged['required'].extend(tmp['required'])
                continue
            if 'properties' in partial_schema:
                merged['properties'].update(partial_schema['properties'])
            if 'required' in partial_schema:
                merged['required'].extend(partial_schema['required'])
        return merged

    def _merge_allof(obj):
        obj_out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'allOf':
                    return _merge_allof(merge(v))
                elif isinstance(v, list):
                    obj_out[k] = [_merge_allof(el) for el in v]
                elif isinstance(v, dict):
                    obj_out[k] = _merge_allof(v)
                else:
                    obj_out[k] = v
            return obj_out
        elif isinstance(obj, list):
            return [_merge_allof(el) for el in obj]
        else:
            return obj

    return _merge_allof(obj)


@dataclass
class ReducedOpenAPISpec:
    servers: List[dict]
    description: str
    endpoints: List[Tuple[str, Union[str, None], dict]]


def reduce_openapi_spec(spec: dict, dereference: bool = True, only_required: bool = True, merge_allof: bool = False) -> ReducedOpenAPISpec:
    """Simplify/distill/minify a spec somehow.

    I want a smaller target for retrieval and (more importantly)
    I want smaller results from retrieval.
    I was hoping https://openapi.tools/ would have some useful bits
    to this end, but doesn't seem so.
    """
    # 1. Consider only get, post, patch, delete endpoints.
    endpoints = [
        (f"{operation_name.upper()} {route}", docs.get("description"), docs)
        for route, operation in spec["paths"].items()
        for operation_name, docs in operation.items()
        if operation_name in ["get", "post", "patch", "delete", "put"]
    ]

    # endpoints = []
    # for route, operation in spec["paths"].items():
    #     for operation_name, docs in operation.items():
    #         if operation_name in ["get", "post", "patch", "delete"]:
    #             if "parameters" in operation:
    #                 if "parameters" in operation:
    #                     docs += operation["parameters"]
    #                 else:
    #                     docs = operation["parameters"]
    #             endpoints.append(
    #                 (f"{operation_name.upper()} {route}", docs.get("description"), docs)
    #             )

    # 2. Replace any refs so that complete docs are retrieved.
    # Note: probably want to do this post-retrieval, it blows up the size of the spec.
    if dereference:
        endpoints = [
            (name, description, dereference_refs(docs, spec))
            for name, description, docs in endpoints
        ]

    # 3. Merge "allof" properties. Maybe very slow.
    if merge_allof:
        endpoints = [
            (name, description, merge_allof_properties(docs))
            for name, description, docs in endpoints
        ]

    # 3. Strip docs down to required request args + happy path response.
    def reduce_endpoint_docs(docs: dict) -> dict:
        out = {}
        if docs.get("description"):
            out["description"] = docs.get("description")
        if docs.get("parameters"):
            if only_required:
                out["parameters"] = [
                    parameter
                    for parameter in docs.get("parameters", [])
                    if parameter.get("required")
                ]
            else:
                out["parameters"] = [
                    parameter
                    for parameter in docs.get("parameters", [])
                ]
        if docs.get("requestBody"):
            out["requestBody"] = docs.get("requestBody")
        if "200" in docs["responses"]:
            out["responses"] = docs["responses"]["200"]
        elif 200 in docs["responses"]:
            out["responses"] = docs["responses"][200]
        return out

    endpoints = [
        (name, description, reduce_endpoint_docs(docs))
        for name, description, docs in endpoints
    ]
    return ReducedOpenAPISpec(
        servers=spec["servers"],
        description=spec["info"].get("description", ""),
        endpoints=endpoints,
    )
