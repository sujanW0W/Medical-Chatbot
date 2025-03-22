import os

from dotenv import load_dotenv


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI


from src.helpers import download_embedding_model
from src.prompts import *

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

load_dotenv()

embedding_model = download_embedding_model()
index_name = "medical-chatbot"


# Load Existing Index

doc_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


retriever = doc_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data['question']
    response = rag_chain.invoke({"input": question})
    print(response)
    return jsonify({'response': response['answer']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
