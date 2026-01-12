import os
import io
import json

from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, TypedDict, Union

from dotenv import load_dotenv
from openai.lib._parsing import type_to_response_format_param
from api.bot.gpt import client


load_dotenv(override=True)


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLM:
    def __init__(self) -> None:
        pass

    def chat(self, messages: list[Message]) -> str:
        pass


class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv('OPENAI_API_KEY'),
        self.client = client
        self.temperature = 0.01
        self.model = model

    def read_prompt(self, prompt_file: str) -> str:
        with open(f'api/bot/Prompts/{prompt_file}', "r", encoding="utf-8") as file:
            return file.read()

    def chat(self, system_message: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def chat_structured(self, messages: list[Message], response_format=BaseModel) -> BaseModel:
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_format,
        )
        return response.choices[0].message.parsed

    def emotion_retriever(self, user_message: str, chat_history: str) -> str:
        return self.chat(system_message="Chat History:" + chat_history + "\n\n" + self.read_prompt("emotion_retriever.md"), user_message=user_message)

    def response_retriever(self, user_message: str, chat_history: str) -> str:
        return self.chat(system_message="Chat History:" + chat_history + "\n\n" + self.read_prompt("response_retriever.md"), user_message=user_message)


class OpenAIBatchILLM(LLM):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY'),
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.requests = []
        self.structures = {}
        self.batch = None

    def chat(self, system_message: str, user_message: str) -> None:
        self.requests.append(
            {
                "custom_id": f"request-{len(self.requests)+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ]
                }
            }
        )

    def chat_structured(self, system_message: str, user_message: str, response_format: BaseModel) -> None:
        self.requests.append(
            {
                "custom_id": f"request-{len(self.requests)+1}-structured",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    "response_format": type_to_response_format_param(response_format),
                }
            }
        )
        self.structures[f"request-{len(self.requests)+1}-structured"] = response_format

    def create(self) -> None:
        f = io.BytesIO()
        for req in self.requests:
            req_json = json.dumps(req).encode('utf-8')
            f.write(req_json)
            f.write(b"\n")
        f.seek(0)

        batch_input_file = self.client.files.create(
            file=f,
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id

        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Eval Job"
            }
        )
        self.batch = batch.id

    def retrieve(self) -> Union[None, list[str]]:
        batch = self.client.batches.retrieve(self.batch)
        if batch.status in ['cancelled', 'cancelling', 'expired', 'failed']:
            raise Exception('Batch failed')
        if batch.status != 'completed':
            return None
        file_response = self.client.files.content(batch.output_file_id)
        output = file_response.text.splitlines()
        output = [json.loads(out) for out in output]
        results = {}
        for out in output:
            if out["response"]["status_code"] != 200:
                continue
            results[out["custom_id"]
                    ] = out["response"]["body"]["choices"][0]["message"]
        output = [v for _, v in sorted(results.items(), key=lambda x: x[0])]
        return output
