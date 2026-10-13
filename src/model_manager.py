from llm_client import client
MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
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
                max_tokens=max_tokens,
                # gpt-oss models count reasoning tokens against max_tokens;
                # keep reasoning short so the actual answer isn't truncated.
                reasoning_effort="low"
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"{model} failed")
            print(e)

            last_error = e
    raise Exception(f"All models failed.\n{last_error}")