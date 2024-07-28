CHAT_TEMPLATE = """
{%- for message in messages %}
    {%- set prefix = '<|begin_of_text|>' if loop.index0==0 else '' %}
    {{- prefix + '<|start_header_id|>'+message['role']+'<|end_header_id|>\n\n' -}}
    {%- if message['role'] == 'assistant' and 'tool_calls' in message %}
        {%- for tool in message['tool_calls'] %}
            {%- set tool_json = {'id': tool['id'], 'name': tool['function']['name'], 'arguments': tool['function']['arguments']} %}
            {{- tool_json }}
        {%- endfor %}
        {{- '<|eot_id|>\n' }}
    {%- else %}
        {{- message['content'] + '<|eot_id|>\n' }}
    {%- endif %}
{%- endfor %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
"""
