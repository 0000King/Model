from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from initialize import *  # Assumes this sets up `documents`, `embeddings`, `llm`, etc.

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["https://model-m46u.onrender.com"])

@app.route("/trackofield_model", methods=["POST"])
def trackofield_model():
    # data = request.get_json()
    # query = data.get("query")
    query="hello there"
    # Initialize Chroma DB
    db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persistent_directory)
    retriever = db.as_retriever(search_kwargs={'k': 200})
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    
    # Get answer from retriever
    retrieved_docs = qa_chain.invoke(query)
    
    # Create a prompt
    prompt = f"Answer this {query} using {retrieved_docs} if you have the answer then give it. Else say I'm not sure."

    # Generate response
    messages = [
        SystemMessage(content="You are a helpful assistant, if you are getting x2, x3 or x times from task summary, then treat it as different tasks, and display it as a different task"),
        HumanMessage(content=prompt)
    ]
    
    result = llm.invoke(messages)
    print(result.content)
    return jsonify(result.content)

@app.route("/home", methods=["GET"])
def home():
    return "Hello"

if __name__ == "__main__":
    app.run(debug=True)