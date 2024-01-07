from flask import Flask, render_template, request, jsonify
import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

app = Flask(__name__, static_url_path='')

os.environ['OPENAI_API_KEY'] = "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"

file_path = '/Users/bihan/Desktop/career/website/bihan_corpus'

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
    app.run(debug=True)