from datasets import load_dataset
from groq import Groq
from dotenv import load_dotenv
from neo4j import GraphDatabase
from transformers import pipeline
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import os
import json
import time
from sentence_transformers import SentenceTransformer, util
import torch


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from neo4j import GraphDatabase

URI = "neo4j+ssc://6eb9f270.databases.neo4j.io"
AUTH = ("neo4j", "7x4XKEgshVbs8DJYqADzDIHn1STM1LLnNZgkkyAyyJY")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

with driver.session() as session:
    result = session.run("RETURN 1")
    print(result.single())


def enrich_with_metadata_groq(query):
    prompt = """
    You are an expert customer support classifier.
    Given a customer support query, return a JSON with:
    - topic: one of [Billing, Login, Technical Issue, Shipping, General Inquiry, Cancellation, Refund, Other]
    - sentiment: Positive, Neutral, or Negative
    - urgency: Low, Medium, or High

    Return only a raw JSON object like this:
    {"topic": "Billing", "sentiment": "Neutral", "urgency": "Low"}

    Do not include code blocks, explanations, or any other text.

    Query: 
    """ + query
    
    try:
        chat_completion = client.chat.completions.create(
            model = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.2,
        )
        content = chat_completion.choices[0].message.content
        print("Raw response:", content) 
        return json.loads(content)
    except Exception as e:
        print("Error - ", e)
        return None

def create_simple_graph_networkx(messages):
    # python .\networkx.py
    G = nx.DiGraph()

    # Add nodes
    for msg in messages:
        G.add_node(msg["id"], label=msg["query"], topic=msg["topic"], sentiment=msg["sentiment"], urgency=msg["urgency"])

    # Add edges — in this example, connect nodes with the same topic
    for i in range(len(messages)):
        for j in range(i + 1, len(messages)):
            if messages[i]["topic"] == messages[j]["topic"]:
                G.add_edge(messages[i]["id"], messages[j]["id"], reason="same_topic")

    # Visualize
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2500, font_size=8)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("Customer Support Knowledge Graph")
    plt.show()

def create_query_and_response(tx, message):
    query = """
    OPTIONAL MATCH (existing:Query {id: $id})
    WITH existing
    WHERE existing IS NULL
    MERGE (t:Topic {name: $topic})
    MERGE (s:Sentiment {type: $sentiment})
    MERGE (u:Urgency {level: $urgency})
    
    CREATE (q:Query {text: $query_text, id: $id})
    CREATE (r:Response {text: $response})
    
    CREATE (q)-[:HAS_RESPONSE]->(r)
    CREATE (q)-[:HAS_TOPIC]->(t)
    CREATE (q)-[:HAS_SENTIMENT]->(s)
    CREATE (q)-[:HAS_URGENCY]->(u)
    """
    tx.run(query,
           id=message["id"],
           query_text=message["query"],
           topic=message["topic"],
           sentiment=message["sentiment"],
           urgency=message["urgency"],
           response=message["response"])
    
def insert_into_neo4j(messages):
    with driver.session() as session:
        for message in messages:
            session.execute_write(
                create_query_and_response,
                message
            )

def similarity_embedding(messages):
    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract only the queries from your messages
    queries = [msg["query"] for msg in messages]

    # Compute embeddings
    embeddings = model.encode(queries, convert_to_tensor=True)

    # Define similarity threshold
    threshold = 0.5

    # Compute pairwise cosine similarity
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

    print("COSINE SCORES : ", cosine_scores)
    # Create SIMILAR_TO relationships in Neo4j
    with driver.session() as session:
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                score = cosine_scores[i][j].item()
                print(score)
                if score > threshold:
                    session.run("""
                        MATCH (q1:Query {id: $id1}), (q2:Query {id: $id2})
                        MERGE (q1)-[:SIMILAR_TO {score: $score}]->(q2)
                    """, id1=i, id2=j, score=score)

def extract_triples_with_llm(query, response):
    """Extract semantic triples for GraphRAG using LLM"""
    prompt = f"""
    Extract semantic triples (subject-predicate-object) from this customer support conversation.
    Focus on problems, solutions, entities, and relationships.
    
    Query: {query}
    Response: {response}
    
    Return a JSON array of triples in this format:
    [
        {{"subject": "Customer", "predicate": "HAS_PROBLEM", "object": "Login_Issue"}},
        {{"subject": "Support", "predicate": "SUGGESTS", "object": "Password_Reset"}},
        {{"subject": "Account", "predicate": "REQUIRES", "object": "Email_Verification"}}
    ]
    
    Extract ALL relevant relationships including:
    - Problems and their types
    - Solutions and recommendations
    - Products and their issues
    - Actions and their outcomes
    - Entities and their properties
    
    Return only valid JSON, no explanations:
    """
    
    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = chat_completion.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error extracting triples: {e}")
        return []

def create_triples_in_neo4j(tx, message, triples):
    """Create semantic triples in Neo4j for GraphRAG"""
    
    # First ensure the Query/Response nodes exist
    query_check = "MATCH (q:Query {id: $id}) RETURN q"
    if not tx.run(query_check, id=message["id"]).single():
        print(f"Query node {message['id']} not found!")
        return
    
    for triple in triples:
        # Create semantic triple
        cypher = """
        MATCH (q:Query {id: $query_id})
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATION {type: $predicate}]->(o)
        
        // Connect to the original query for context
        MERGE (q)-[:CONTAINS_TRIPLE]->(s)
        MERGE (q)-[:CONTAINS_TRIPLE]->(o)
        """
        
        tx.run(cypher, 
               query_id=message["id"],
               subject=triple["subject"],
               predicate=triple["predicate"],
               object=triple["object"])

def build_graphrag_knowledge_graph(messages):
    """Build a proper knowledge graph for GraphRAG"""
    print("Building GraphRAG knowledge graph...")
    
    for i, message in enumerate(messages):  # Start with first 10
        print(f"\n=== Processing Message {i} ===")
        print(f"Query: {message['query']}")
        print(f"Response: {message['response'][:100]}...")
        
        # Extract semantic triples
        triples = extract_triples_with_llm(message['query'], message['response'])
        print(f"Extracted triples: {triples}")
        
        if triples:
            # Insert into Neo4j
            with driver.session() as session:
                try:
                    session.execute_write(create_triples_in_neo4j, message, triples)
                    print(f"✅ Created {len(triples)} triples")
                except Exception as e:
                    print(f"❌ Error creating triples: {e}")
        else:
            print("No triples extracted")
        
        time.sleep(1)  # Rate limiting for API

dataset = load_dataset("Kaludi/Customer-Support-Responses")

messages  = []

for i in range(len(dataset["train"])):
    query = dataset["train"][i]["query"]
    metadata = enrich_with_metadata_groq(query)
    print(f"\nQuery: {query}")
    print("Metadata:", metadata)
    message = {
        "id": i,
        "query": query,
        "response": dataset["train"][i]["response"],
        "topic": metadata["topic"],
        "sentiment": metadata["sentiment"],
        "urgency": metadata["urgency"]
    }
    print(message)
    messages.append(message)
    
#create_simple_graph_networkx(messages)
#insert_into_neo4j(messages)
#similarity_embedding(messages)
build_graphrag_knowledge_graph(messages)

