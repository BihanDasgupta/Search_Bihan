from flask import Flask, render_template, request, jsonify
import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OpenAI API key not found in environment variables. Reset it in the .env file.")


file_path = 'bihan_corpus'

documents = SimpleDirectoryReader(file_path).load_data()

index = GPTVectorStoreIndex(documents)

query_engine = index.as_query_engine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    response = query_engine.query(query)

    return render_template('result.html', query=query, response=response)

if __name__ == '__main__':
    app.run()