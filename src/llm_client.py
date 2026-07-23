from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print(api_key)      # Temporary

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)