import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, List, Dict
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI()

os.environ['LANGCHAIN_PROJECT'] = 'ConditionalChain LLM App'

# we can use the os.environ to set the new project details in Langsmith


class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="Sentiment for the feedback")


parser1 = PydanticOutputParser(pydantic_object=Sentiment)

prompt1 = PromptTemplate(
    template="generate the sentiment for the given input:{input}\n {format_instructions}",
    input_variables=['input'],
    partial_variables={
        'format_instructions': parser1.get_format_instructions()}
)
parser2 = StrOutputParser()
p2 = PromptTemplate(
    template="write appropriate response for postive feedback{feedback}",
    input_variables=['feedback']
)

p3 = PromptTemplate(
    template="write appropriate response for negative feedback{feedback}",
    input_variables=['feedback']
)

feedbackChain = RunnableSequence(prompt1, model, parser1)

branchChain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', RunnableSequence(p2, model, parser2)),
    (lambda x: x.sentiment == 'negative', RunnableSequence(p3, model, parser2)),
    RunnableLambda(lambda x: "could not find sentiment")
)

finalchain = RunnableSequence(feedbackChain, branchChain)

result = finalchain.invoke({'input': " i don't like this mobile"})

print(result)
