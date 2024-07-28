from threading import Thread

import litserve as ls
import torch
from litserve.specs.openai import ChatCompletionRequest, ChatMessage
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from config import CHAT_TEMPLATE
from utils import extract_tool_calls_from_buffer, get_tools_prefix_messages


class OpenAISpecLitAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device

        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # load chat template for "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.tokenizer.chat_template = CHAT_TEMPLATE

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quantization_config
        )

        # extract special tokens
        self.special_tokens = [
            item
            for value in self.tokenizer.special_tokens_map.values()
            for item in (value if isinstance(value, list) else [value])
        ]

        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

    def decode_request(self, request: ChatCompletionRequest, context):
        # set values for temperature, max_tokens, and top_p
        context["temperature"] = request.temperature
        context["max_tokens"] = request.max_tokens if request.max_tokens else 1024
        context["top_p"] = request.top_p

        messages = get_tools_prefix_messages(request.messages, request.tools)

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        return model_inputs

    def predict(self, model_inputs, context):
        generation_kwargs = dict(
            **model_inputs,
            streamer=self.streamer,
            max_new_tokens=context["max_tokens"],
            temperature=context["temperature"],
            top_p=context["top_p"],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for text in self.streamer:
            yield text

    def encode_response(self, output_generator) -> ChatMessage:  # type: ignore
        buffer = []
        for output in output_generator:
            buffer.append(output)
            print(output, end="", flush=True)
            # check if tool calls
            if "".join(buffer).startswith("<function"):
                tool_calls = extract_tool_calls_from_buffer(buffer)
                yield ChatMessage(role="assistant", content="", tool_calls=tool_calls)
                continue

            if self.tokenizer.eos_token in output:
                output = output.replace(self.tokenizer.eos_token, "")

            yield ChatMessage(role="assistant", content=output)


if __name__ == "__main__":
    server = ls.LitServer(OpenAISpecLitAPI(), spec=ls.OpenAISpec())
    server.run(port=8000)
