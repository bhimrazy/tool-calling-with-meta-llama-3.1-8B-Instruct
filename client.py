# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import requests

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the orginal use question.",
    },
    {"role": "user", "content": "What's the weather like in Kathmandu today?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_6duDxk",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": '{"location": "Kathmandu, NP", "unit": "celsius"}',
                },
            }
        ],
    },
    {
        "role": "ipython",
        "tool_call_id": "call_6duDxk",
        "name": "get_current_weather",
        "content": '{"location": "Kathmandu", "temperature": "32", "unit": "celsius"}',
    },
]

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "stream": False,
        "temperature": 0.5,
        "max_tokens": 1024,
        "top_p": 0.95,
        "tools": tools,
        "messages": messages,
    },
)

completion = response.json()
assistant_message = completion["choices"][0]["message"]
print(
    f"Status: {response.status_code}\nResponse:\n {json.dumps(assistant_message, indent=4)}"
)
