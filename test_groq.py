import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key and initialize the Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY") 
)

# Test the client
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    model=os.getenv("LLM_MODEL"),
)

print(chat_completion.choices[0].message.content)