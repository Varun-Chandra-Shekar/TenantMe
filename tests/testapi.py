from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()                      # reads ANTHROPIC_API_KEY from .env
client = Anthropic()               # picks up the key from the environment

response = client.messages.create(
    model="claude-haiku-4-5-20251001",   # your cheap dev model
    max_tokens=10,                        # cap output — keeps it tiny and cheap
    messages=[{"role": "user", "content": "Reply with just: OK"}],
)

print(response.content[0].text)
print("Tokens — in:", response.usage.input_tokens,
      "out:", response.usage.output_tokens)
