import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain.schema import StrOutputParser
from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap


neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="review_text_embedding_index",
    node_label="Review",
    text_node_property="text",
    embedding_node_property="textEmbedding",
    retrieval_query="""
RETURN 
node.text as text,
score,
  {
    reviewId: node.id,
    stars: node.stars,
    date: node.date
  } AS metadata
    """
)

retriever = neo4jvector.as_retriever()

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
review_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

def search_similar_question(question):
    """Searching for similar questions"""
    print("Executing embeddings search")
    return review_retriever.invoke({"input": question})

def p(output):
    print(output)
    return {"result": output}