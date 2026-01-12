import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)


def openai_req_generator(system_prompt, user_prompt=None, json_output=False, temperature=0.01):
    messages = [{"role": "system", "content": system_prompt}]
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    if json_output:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            # model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    else:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            # model="chatgpt-4o-latest",
            temperature=temperature,
        )
    return chat_completion.choices[0].message.content

# New function for passing full message history


def openai_req_with_history(messages, temperature=0.01):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        temperature=temperature,
    )
    return chat_completion.choices[0].message.content
