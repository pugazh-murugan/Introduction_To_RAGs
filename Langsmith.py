from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langsmith import traceable
from openai import OpenAI
from langsmith.wrappers import wrap_openai

load_dotenv()

LANGSMITH_TRACING  = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")


@traceable
def make_greeting(name):
    return f"Hello {name}, welcome to GenAI!"

# wrap_openai is the ONLY change needed for tracking.
client = wrap_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "Say hello in one word."}],
)

print(response.choices[0].message.content)

# This call is now recorded in LangSmith as its own step.
print(make_greeting("Pugazh Murugan"))

prompt = ChatPromptTemplate.from_template("Explain {topic} in one simple sentence.")
model = ChatOpenAI(model="gpt-5.4-mini")
chain = prompt | model | StrOutputParser()

# This run is automatically recorded in LangSmith.
print(chain.invoke({"topic": "LangSmith"}))