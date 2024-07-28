# adapted from: https://github.com/meta-llama/llama-agentic-system/blob/main/llama_agentic_system/system_prompt.py
from datetime import datetime
from typing import List
import json
import re
import secrets
import string


from litserve.specs.openai import ChatMessage, Function, Tool


def get_tools_prefix_messages(
    messages: List[ChatMessage], custom_tools: List[Tool] = None
) -> List[ChatMessage]:
    messages = messages.copy()
    content = ""
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d %B %Y")
    date_str = f"""
Cutting Knowledge Date: December 2023
Today Date: {formatted_date}\n\n"""
    content += date_str

    if custom_tools:
        tools_prompt = get_system_prompt_for_custom_tools(custom_tools)
        content += tools_prompt

    if messages[0].role != "system":
        content += "You are a helpful Assistant."
        messages.insert(0, ChatMessage(role="system", content=content))
    else:
        content += messages[0].content
        messages[0].content = content

    return messages


def get_system_prompt_for_custom_tools(custom_tools: List[Tool]) -> str:
    custom_tool_params = ""
    for tool in custom_tools:
        custom_tool_params += get_instruction_string(tool.function) + "\n"
        custom_tool_params += tool.model_dump_json() + "\n\n"

    content = f"""
You have access to the following functions:

{custom_tool_params}
Think very carefully before calling functions.
If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- If looking for real time information use relevant functions before falling back to brave_search
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

"""
    return content


def get_instruction_string(custom_tool_definition: Function) -> str:
    return f"Use the function '{custom_tool_definition.name}' to '{custom_tool_definition.description}'"


CUSTOM_TOOL_CALL_PATTERN = re.compile(
    r"<function=(?P<function_name>[^}]+)>(?P<args>{.*?})"
)


def generate_call_id():
    """
    Generate a unique call ID starting with 'call_' followed by exactly 9 characters in the format a-z, A-Z, 0-9.

    Returns:
        str: A unique call ID.
    """
    characters = string.ascii_letters + string.digits
    unique_id = "call_" + "".join(secrets.choice(characters) for _ in range(6))
    return unique_id


def extract_tool_calls_from_buffer(buffer):
    """
    Extract tool calls from the given buffer.

    Adapted from: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/tool_utils.py#L49

    Args:
        buffer (list): A list of strings containing the buffer data.

    Returns:
        list: A list of formatted tool calls or None if no tool calls found.
    """
    joined_buffer = "".join(buffer)
    match = re.search(CUSTOM_TOOL_CALL_PATTERN, joined_buffer)
    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            tool_calls = [
                {
                    "id": generate_call_id(),
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(args),
                    },
                    "type": "function",
                }
            ]
            return tool_calls
        except json.JSONDecodeError as error:
            print(f"Error parsing function arguments: {error}")
            return None
    return None
