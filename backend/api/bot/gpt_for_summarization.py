import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)


def openai_req_generator(system_prompt, json_output=False, temperature=0.01):
    if json_output:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            model="gpt-4o",
            # model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    else:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            model="gpt-4o",
            # model="chatgpt-4o-latest",
            temperature=temperature,
        )
    return chat_completion.choices[0].message.content
