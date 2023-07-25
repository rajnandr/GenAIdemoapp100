import os
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from PIL import Image
import time

image = Image.open('Wipro logo.png')
st.image(image)

st.title("Wipro impact | The inquisitive sustainability leader")  

st.header("Learn some of the best practices in sustainability from success stories of leading companies.. ")


st.subheader("Welcome!. Today, What company's sustainability story is inspiring you ?.. ")


os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2023-03-15-preview'

llmgpt3 = AzureOpenAI(      deployment_name="testdavanci", model_name="text-davinci-003" )
#llmchatgpt = AzureOpenAI(     deployment_name="esujnand", model_name="gpt-35-turbo" )


with st.form("my_form"):

   myurl = st.text_input("What is the URL of the sustainability report?", "https://www.wipro.com/content/dam/nexus/en/sustainability/sustainability_reports/wipro-sustainability-report-fy-2021-22.pdf")

   yourquestion = st.text_input('Ask your question on best practices', 'What is Wipro plans for Biodiversity in 2024?')
   st.write('Your input is ', yourquestion)

   # Every form must have a submit button.
   submitted = st.form_submit_button("Ask question")
   if submitted:
      st.write("AI is looking for the answer...It will take atleast 2 mintutes... Answers will appear below....")

      if myurl:
          index = None
          loader1 = PyPDFLoader(myurl)
          langchainembeddings = OpenAIEmbeddings(deployment="textembedding", chunk_size=1)

          index = VectorstoreIndexCreator(
                  # split the documents into chunks
                  text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
                  # select which embeddings we want to use
                  embedding=langchainembeddings,
                  # use Chroma as the vectorestore to index and search embeddings
                  vectorstore_cls=Chroma
              ).from_loaders([loader1])
              
          st.write("indexed PDF...AI finding answer....please wait")



      if yourquestion:
        answer = index.query(llm=llmgpt3, question=yourquestion, chain_type="map_reduce")
        st.subheader(answer)

