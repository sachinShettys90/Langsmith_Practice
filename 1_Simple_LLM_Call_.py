from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatOpenAI()
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of India?"})
print(result)


# Since we used langsmith , we can have a trace , Latency , Stats , TotalTokens
# within LANGCHAIN_PROJECT="langsmith-demo" created in .env file

# NOTE : here we used PromptTemplate , ChatOpenAI call , StrOutputParcer, this entire inputs and their outputs will be captured in Langsmith as individual component
