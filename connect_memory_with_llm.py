"""
connect_memory_with_llm.py

Connects a FAISS vector database with a Groq LLM using LangChain for retrieval-augmented QA.
"""

import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Setup LLM (Mistral with HuggingFace)
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

def load_llm(model_name, groq_api_key):
    """Load the Groq LLM with specified parameters."""
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.5,
        groq_api_key=groq_api_key,
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    """Create a PromptTemplate for the QA chain."""
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    # Load Database
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(GROQ_MODEL, GROQ_API_KEY),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    # Now invoke with a single query
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})
    print("RESULT: ", response["result"])
    print("SOURCE DOCUMENTS: ", response["source_documents"])

if __name__ == "__main__":
    main()