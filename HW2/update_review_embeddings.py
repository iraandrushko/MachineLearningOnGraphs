from neo4j import GraphDatabase
import streamlit as st 
from build_review_embeddings import run_query


uri = st.secrets["NEO4J_URI"]
username = st.secrets["NEO4J_USERNAME"]
password = st.secrets["NEO4J_PASSWORD"]

load_embeddings_query = """
CALL apoc.periodic.iterate(
  "
  LOAD CSV WITH HEADERS
  FROM 'https://www.dropbox.com/scl/fi/4gu29g0st1vwixgioay0j/reviews.csv?rlkey=o3xca5fr9uzj0av8ec7xonm9e&st=6068y0wf&dl=1'
  AS row
  RETURN row
  ",
  "
  MATCH (r:Review {id: row.id})
  CALL db.create.setVectorProperty(r, 'textEmbedding', apoc.convert.fromJsonList(row.embedding))
  YIELD node
  RETURN count(*)
  ",
  {batchSize: 512, parallel: true}
)

"""

drop_index = """
DROP INDEX review_text_embedding_index IF EXISTS
"""
create_vector_index_query = """
CREATE VECTOR INDEX review_text_embedding_index IF NOT EXISTS
FOR (r:Review)
ON r.textEmbedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
"""

if __name__ == "__main__":
    driver = GraphDatabase.driver(uri, auth=(username, password))

    st.write("Running embedding load query...")
    # result1 = run_query(driver, load_embeddings_query)
    # st.success(f"Embeddings loaded: {result1}")
    run_query(driver, drop_index)
    st.write("Creating vector index...")
    result2 = run_query(driver, create_vector_index_query)
    driver.close()
    print("end")
