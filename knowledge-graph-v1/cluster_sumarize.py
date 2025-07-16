from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
# --- CONFIG ---
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama3-70b-8192"

# --- LLM Setup ---
llm = ChatGroq(model=LLM_MODEL)

URI = "neo4j+ssc://6eb9f270.databases.neo4j.io"
AUTH = ("neo4j", "7x4XKEgshVbs8DJYqADzDIHn1STM1LLnNZgkkyAyyJY")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

# --- LangChain Summary Function ---
def summarize_cluster(entities, queries):
    prompt = f"""
            You are analyzing a cluster of related customer support queries.

            Entities in this cluster:
            {chr(10).join(f"- {e}" for e in entities)}

            Example queries from this cluster:
            {chr(10).join(f"- {q}" for q in queries)}

            Summarize what this cluster is about in 1–3 sentences.
            """
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

# --- Step 1: Get all communityIds ---
def get_community_ids(tx):
    result = tx.run("MATCH (e:Entity) RETURN DISTINCT e.communityId AS id")
    return [record["id"] for record in result]

# --- Step 2: Get sample entities and queries for each community ---
def get_entities_and_queries(tx, community_id, entity_limit=8, query_limit=5):
    entity_res = tx.run("""
        MATCH (e:Entity)
        WHERE e.communityId = $cid
        RETURN e.name AS name LIMIT $limit
        """, cid=community_id, limit=entity_limit)
    entities = [record["name"] for record in entity_res]

    query_res = tx.run("""
        MATCH (q:Query)-[:CONTAINS_TRIPLE]->(e:Entity)
        WHERE e.communityId = $cid
        RETURN DISTINCT q.text AS text
        LIMIT $limit
        """, cid=community_id, limit=query_limit)
    queries = [record["text"] for record in query_res]

    return entities, queries

# --- Step 3: Store cluster summary ---
def save_summary(tx, community_id, summary):
    tx.run("""
        MERGE (c:Cluster {id: $cid})
        SET c.summary = $summary
        """, cid=community_id, summary=summary)

# --- Main loop ---
with driver.session() as session:
    community_ids = session.read_transaction(get_community_ids)
    print(f"Found {len(community_ids)} communities.")

    for cid in community_ids:
        print(f"Processing community {cid}...")
        entities, queries = session.read_transaction(get_entities_and_queries, cid)
        if not entities or not queries:
            print(f"Skipping community {cid} (not enough data).")
            continue

        summary = summarize_cluster(entities, queries)
        print(f"Summary:\n{summary}\n")
        session.write_transaction(save_summary, cid, summary)
