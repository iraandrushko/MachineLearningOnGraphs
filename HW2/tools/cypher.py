import streamlit as st
from llm import llm
from graph import graph
from langchain.schema import StrOutputParser
from operator import itemgetter
from langchain_neo4j import GraphCypherQAChain
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions on yelp reviews.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.


Examples:
show me 25 users
MATCH (n:User) RETURN n LIMIT 25

show me reviews relation 
MATCH p=()-[r:REVIEWS]->() RETURN p LIMIT 25

To find local businesses in a specific city:
MATCH (b:Business)-[:IN_CITY]->(c:City {name: "City Name"}) RETURN b.name, b.address, b.rating

Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_prompt,
)

get_yelp_response = (
    cypher_qa 
    | RunnableLambda(lambda output: "For query:" + output.get('query') + "we found objects: " + output.get('result', "None"))
)
