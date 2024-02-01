import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

loader = CSVLoader(file_path="Bihan_Corpus.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

template2 = """
"
You are to answer any question the user inputs. You can answer general questions, but also questions that are specific to Bihan Dasgupta. You have extensive knowledge about Bihan from the data fed to you below. When referring to Bihan, please use pronouns she/her/hers.  
Question: {question}

Data about Bihan: {relevant_data}

"
"""

prompt = PromptTemplate(
    input_variables=["question", "relevant_data"],
    template=template2
)

chain = LLMChain(llm=llm, prompt=prompt)


def generate_response(question):
    relevant_data = retrieve_info(question)
    response = chain.run(question=question, relevant_data=relevant_data)
    return response


def main():
    st.set_page_config(
        page_title="Get to know me", page_icon=":female-technologist:")

    # announcement = "(P.S. This was built in 3 days, imagine what I can do in 30 :sunglasses:)"
    # st.toast(body=announcement)
    # st.balloons()

    message = st.text_area("Hi, I am Bihan Dasgupta. Ask me any questions you want to know about me.")

    if message:
        st.write("Typing...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()