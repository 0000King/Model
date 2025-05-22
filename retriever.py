import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage , AIMessage
from initialize import *

def trackofield_model(query):
    #Final Question Answering
    db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persistent_directory)
    retriever = db.as_retriever(search_kwargs={'k': 200})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    #get retrived docs 
    retreived_docs=qa_chain.invoke(query)
    prompt=f"Answer this {query} using {retreived_docs} if you have the answer then give it .else say i'm not sure"

    message=[
        SystemMessage("You are a helpful assistant"),
        HumanMessage(content=prompt)
    ]
    result=llm.invoke(message)
    return result.content



# , if you are getting x2, x3 or x times from task summary, then treat it as different tasks, and display it as a different task, and NBD's full form is New Business Deal