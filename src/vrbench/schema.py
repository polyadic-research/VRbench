import json
from importlib.resources import files
from jsonschema import Draft202012Validator


def validate(name,value):
    schema=json.loads(files('vrbench').joinpath('resources/schemas/'+name+'.schema.json').read_text(encoding='utf8'))
    Draft202012Validator(schema).validate(value)
