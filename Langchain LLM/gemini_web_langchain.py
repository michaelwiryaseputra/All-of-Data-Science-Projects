from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

#genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")

#Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")

#Load the blog
loader = WebBaseLoader("https://thenewstack.io/the-building-blocks-of-llms-vectors-tokens-and-embeddings/")
docs = loader.load()

#Define the Summarize Chain
template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

#Invoke Chain
response=stuff_chain.invoke(docs)
print(response["output_text"])