#!/usr/bin/env python3
import os, json
from openai import OpenAI
 
# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT = """
Write a prompt here and fill in data using .format. For example, if you want to interpolate data into a prompt with a variable called "data".
use 

prompt = PROMPT.format(data=data)

This will substitute the variable contents into instances with the following construct: {data}
"""

prompt = PROMPT.format(data=data)
model = "gpt-4o-mini"

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an expert at analyzing scientific queries about NGS diagnostics."},
        {"role": "user", "content": prompt}
    ],  
    response_format={"type": "json_object"},
    temperature=0.3  # Lower temperature for more consistent decomposition
)   

result = json.loads(str(response.choices[0].message.content))

# Add original data to result
result["original_data"] = data

print(result)

