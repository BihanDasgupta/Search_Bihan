from flask import Flask, render_template, request, jsonify
import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

app = Flask(__name__, static_url_path='')

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

# Define the path to the directory containing your documents
file_path = '/Users/bihan/Desktop/career/website/bihan_corpus'

# Load the documents from the specified directory
documents = SimpleDirectoryReader(file_path).load_data()

# Create an index using the GPTVectorStoreIndex with the loaded documents
index = GPTVectorStoreIndex(documents)

# Convert the index to a query engine
query_engine = index.as_query_engine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    # Use the query engine to search for documents related to the user's input
    response = query_engine.query(query)

    return render_template('result.html', query=query, response=response)

if __name__ == '__main__':
    app.run(debug=True)