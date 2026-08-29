from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate the joke for the given input:{input}",
    input_variables=['input']
)

prompt2 = PromptTemplate(
    template="Give me the explanation for the generated joke:{joke}",
    input_variables=['joke']
)


joke_generatorChain = RunnableSequence(prompt1, model, parser)

parallelchain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

finalChain = RunnableSequence(joke_generatorChain, parallelchain)

result = finalChain.invoke({'input': "Galaxy"})
print(result)
