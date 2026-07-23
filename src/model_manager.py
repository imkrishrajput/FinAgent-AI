from llm_client import client
MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]


def chat(messages, temperature=0, max_tokens=8192):

    last_error = None

    for model in MODELS:

        print(f"Trying model: {model}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"{model} failed")
            print(e)

            last_error = e
    raise Exception(f"All models failed.\n{last_error}")