from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_openai import ChatOpenAI


model = ChatOpenAI(model="gpt-5")
response = model("What is the capital of France?")
print(response.content)

