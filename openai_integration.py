import openai
from openai import OpenAI

client = OpenAI()

def generate_openai_response(messages, model="gpt-4o", temperature=0.1, max_tokens=1000):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from OpenAI: {e}"
