from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
from typing import List, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import logging
import numpy as np
import faiss
import json
import pickle
from pathlib import Path
from groq import Groq

load_dotenv()

# === CONFIG ===
URI = "neo4j+ssc://6eb9f270.databases.neo4j.io"
AUTH = ("neo4j", "7x4XKEgshVbs8DJYqADzDIHn1STM1LLnNZgkkyAyyJY")

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize Neo4j driver
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    
# === Initialize free embedding model (sentence-transformers)
# This model is downloaded locally and runs for free
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, 384-dimensional embeddings

# === Initialize Groq LLM for text generation
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Knowledge Graph Query API", version="1.0.0")

# Add CORS middleware - THIS IS THE KEY FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FAISS Vector Store Implementation ---
class FAISSVectorStore:
    def __init__(self, store_path: str = VECTOR_STORE_PATH, embedding_dim: int = 384):
        self.store_path = Path(store_path)
        self.store_path.mkdir(exist_ok=True)
        self.embedding_dim = embedding_dim
        
        # FAISS index for similarity search
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
        
        # Metadata storage
        self.metadata_file = self.store_path / "metadata.json"
        self.index_file = self.store_path / "faiss.index"
        
        # Load existing index if available
        self.community_ids = []
        self.metadata = {}
        self._load_index()
        
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.community_ids = data.get('community_ids', [])
                    self.metadata = data.get('metadata', {})
                logger.info(f"Loaded metadata for {len(self.community_ids)} communities")
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self._initialize_empty_index()
    
    def _initialize_empty_index(self):
        """Initialize empty index"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.community_ids = []
        self.metadata = {}
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            faiss.write_index(self.index, str(self.index_file))
            
            data = {
                'community_ids': self.community_ids,
                'metadata': self.metadata
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("Successfully saved index and metadata")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def add_embeddings(self, community_id: str, embedding: List[float], metadata: dict = None):
        """Add embeddings to FAISS vector store"""
        try:
            # Normalize embedding for cosine similarity
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(embedding_array)
            
            # Add to FAISS index
            self.index.add(embedding_array)
            
            # Store metadata
            self.community_ids.append(community_id)
            self.metadata[community_id] = metadata or {}
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Added embedding for community {community_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            return False
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[tuple]:
        """
        Search FAISS index for similar embeddings
        Returns list of tuples [(community_id, similarity_score), ...]
        """
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_array)
            
            # Search FAISS index
            scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            # Return results with community IDs and scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.community_ids):
                    community_id = self.community_ids[idx]
                    results.append((community_id, float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []
    
    def get_stats(self):
        """Get vector store statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "total_communities": len(self.community_ids)
        }

# Initialize vector store
vector_store = FAISSVectorStore()

# Pydantic models for request/response
class QueryDetail(BaseModel):
    query_text: str
    response_text: Optional[str] = None
    query_id: Optional[str] = None

class ClusterDetailResponse(BaseModel):
    cluster_id: int
    summary: str
    entity_count: int
    top_entities: List[str]
    queries: List[QueryDetail]
    total_queries: int
    
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class ClusterInfo(BaseModel):
    communityId: int
    summary: str
    related_queries: List[str]
    entity_count: Optional[int] = 0
    top_entities: Optional[List[str]] = []
    similarity_score: Optional[float] = 0.0

class QueryResponse(BaseModel):
    answer: str
    clusters: List[ClusterInfo]
    query_embedding_dim: Optional[int] = None

class IndexRequest(BaseModel):
    community_id: int
    force_reindex: Optional[bool] = False

class IndexResponse(BaseModel):
    message: str
    indexed_communities: List[int]

# Neo4j transaction functions
def fetch_cluster_info(tx, community_id: int) -> ClusterInfo:
    """Fetch detailed cluster information from Neo4j"""
    cypher = """
    MATCH (c:Entity {communityId: $community_id})
    OPTIONAL MATCH (q:Query)-[:CONTAINS_TRIPLE]->(c)
    OPTIONAL MATCH (cluster:Cluster {id: $community_id})
    WITH c.communityId AS communityId, 
         cluster.summary AS summary,
         collect(DISTINCT q.text)[0..5] AS related_queries,
         count(DISTINCT c) AS entity_count,
         collect(DISTINCT c.name)[0..10] AS top_entities
    RETURN communityId, summary, related_queries, entity_count, top_entities
    LIMIT 1
    """
    result = tx.run(cypher, community_id=community_id)
    record = result.single()
    
    if record:
        return ClusterInfo(
            communityId=record["communityId"],
            summary=record["summary"] or "",
            related_queries=record["related_queries"] or [],
            entity_count=record["entity_count"] or 0,
            top_entities=record["top_entities"] or []
        )
    else:
        return ClusterInfo(
            communityId=community_id, 
            summary="", 
            related_queries=[],
            entity_count=0,
            top_entities=[]
        )

def get_all_communities(tx) -> List[int]:
    """Get all community IDs from Neo4j"""
    cypher = """
    MATCH (e:Entity)
    WHERE e.communityId IS NOT NULL
    RETURN DISTINCT e.communityId AS communityId
    ORDER BY communityId
    """
    result = tx.run(cypher)
    return [record["communityId"] for record in result]

def get_community_summary(tx, community_id: int) -> str:
    """Get community summary from Cluster nodes (FIXED)"""
    cypher = """
    MATCH (c:Cluster {id: $community_id})
    RETURN c.summary AS summary
    """
    result = tx.run(cypher, community_id=community_id)
    record = result.single()
    
    if record and record["summary"]:
        return record["summary"]
    
    # Fallback: Generate summary if not exists
    cypher_entities = """
    MATCH (e:Entity {communityId: $community_id})
    RETURN collect(e.name)[0..20] AS entities,
           count(e) AS entity_count
    """
    result = tx.run(cypher_entities, community_id=community_id)
    record = result.single()
    
    if record:
        entities = record["entities"] or []
        entity_count = record["entity_count"] or 0
        if entities:
            return f"Community {community_id} contains {entity_count} entities including: {', '.join(entities[:10])}"
    
    return f"Community {community_id} - No entities found"

def update_community_summary(tx, community_id: int, summary: str):
    """Update community summary in Neo4j (separate write transaction)"""
    cypher = """
    MERGE (c:Cluster {id: $community_id})
    SET c.summary = $summary
    """
    tx.run(cypher, community_id=community_id, summary=summary)

def generate_answer_with_groq(query: str, context: str) -> str:
    """Generate answer using Groq LLM"""
    try:
        prompt = f"""User query: {query}

                Context from knowledge graph:
                {context}

                Based on the user query and the context above, provide a helpful and concise answer. 
                If the information is not sufficient to answer the query, mention what additional information might be needed.
                Keep the response focused and informative.
                """

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Free tier model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on knowledge graph data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating answer with Groq: {e}")
        return f"I found relevant information but couldn't generate a response due to an error: {str(e)}"

def fetch_cluster_queries_and_responses(tx, community_id: int) -> dict:
    """Fetch all queries and responses for a specific cluster"""
    cypher = """
    MATCH (e:Entity {communityId: $community_id})
    OPTIONAL MATCH (q:Query)-[:CONTAINS_TRIPLE]->(e)
    OPTIONAL MATCH (q)-[:HAS_RESPONSE]->(r:Response)
    OPTIONAL MATCH (cluster:Cluster {id: $community_id})
    WITH cluster.summary AS summary,
         count(DISTINCT e) AS entity_count,
         collect(DISTINCT e.name)[0..20] AS top_entities,
         collect(DISTINCT {
             query_text: q.text,
             response_text: r.text,
             query_id: q.id
         }) AS query_response_pairs
    RETURN summary, entity_count, top_entities, query_response_pairs
    """
    result = tx.run(cypher, community_id=community_id)
    record = result.single()
    
    if record:
        # Filter out null queries and deduplicate
        queries = []
        seen_queries = set()
        for item in record["query_response_pairs"]:
            if item["query_text"] and item["query_text"] not in seen_queries:
                queries.append(QueryDetail(
                    query_text=item["query_text"],
                    response_text=item["response_text"],
                    query_id=item["query_id"]
                ))
                seen_queries.add(item["query_text"])
        
        return {
            "summary": record["summary"] or "",
            "entity_count": record["entity_count"] or 0,
            "top_entities": record["top_entities"] or [],
            "queries": queries,
            "total_queries": len(queries)
        }
    
    return {
        "summary": "",
        "entity_count": 0,
        "top_entities": [],
        "queries": [],
        "total_queries": 0
    }

def search_clusters_by_content(tx, search_term: str, limit: int = 10) -> List[dict]:
    """Search for clusters that contain specific content in queries/responses"""
    cypher = """
    MATCH (e:Entity)
    WHERE e.communityId IS NOT NULL
    OPTIONAL MATCH (q:Query)-[:CONTAINS_TRIPLE]->(e)
    OPTIONAL MATCH (q)-[:HAS_RESPONSE]->(r:Response)
    WHERE toLower(q.text) CONTAINS toLower($search_term) 
       OR toLower(r.text) CONTAINS toLower($search_term)
    WITH e.communityId AS communityId, 
         count(DISTINCT q) AS matching_queries,
         collect(DISTINCT q.text)[0..3] AS sample_queries
    WHERE matching_queries > 0
    RETURN communityId, matching_queries, sample_queries
    ORDER BY matching_queries DESC
    LIMIT $limit
    """
    result = tx.run(cypher, search_term=search_term, limit=limit)
    return [dict(record) for record in result]

# Enhanced query processing function
def process_query_intent(query: str) -> dict:
    """Analyze query intent to determine if it's asking for specific cluster data"""
    query_lower = query.lower()
    
    # Check for cluster-specific queries
    import re
    cluster_pattern = r'cluster\s+(\d+)|community\s+(\d+)'
    match = re.search(cluster_pattern, query_lower)
    
    if match:
        cluster_id = int(match.group(1) or match.group(2))
        
        # Determine what they want from the cluster
        if any(word in query_lower for word in ['queries', 'query', 'questions', 'responses', 'answers']):
            return {
                "type": "cluster_detail",
                "cluster_id": cluster_id,
                "wants_queries": True
            }
        else:
            return {
                "type": "cluster_summary",
                "cluster_id": cluster_id,
                "wants_queries": False
            }
    
    # Check for content-based search
    if any(word in query_lower for word in ['find', 'search', 'contains', 'about']):
        return {
            "type": "content_search",
            "search_term": query
        }
    
    # Default to semantic search
    return {
        "type": "semantic_search",
        "query": query
    }

# API Routes
@app.get("/")
def read_root():
    return {
        "message": "Knowledge Graph Query API", 
        "version": "1.0.0",
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_store": "FAISS",
        "llm": "Groq Llama3"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        
        vector_stats = vector_store.get_stats()
        
        return {
            "status": "healthy", 
            "neo4j": "connected",
            "vector_store": vector_stats
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/query", response_model=QueryResponse)
def query_clusters(request: QueryRequest):
    """Enhanced query endpoint - handles both semantic search and cluster-specific queries"""
    user_query = request.query
    top_k = request.top_k

    try:
        # Step 1: Analyze query intent
        intent = process_query_intent(user_query)
        
        # Step 2: Handle different query types
        if intent["type"] == "cluster_detail":
            # Direct cluster query - get specific cluster info
            cluster_id = intent["cluster_id"]
            
            with driver.session() as session:
                cluster_data = session.read_transaction(fetch_cluster_queries_and_responses, cluster_id)
            
            if cluster_data["entity_count"] == 0:
                return QueryResponse(
                    answer=f"Cluster {cluster_id} not found in the knowledge graph.",
                    clusters=[],
                    query_embedding_dim=0
                )
            
            # Format the response with queries and responses
            queries_text = []
            for i, query_detail in enumerate(cluster_data["queries"][:10], 1):  # Show first 10
                query_text = f"{i}. Query: {query_detail.query_text}"
                if query_detail.response_text:
                    query_text += f"\n   Response: {query_detail.response_text}"
                queries_text.append(query_text)
            
            queries_summary = "\n\n".join(queries_text) if queries_text else "No queries found in this cluster."
            
            answer = f"""Cluster {cluster_id} Information:
- Summary: {cluster_data['summary']}
- Total Entities: {cluster_data['entity_count']}
- Total Queries: {cluster_data['total_queries']}
- Top Entities: {', '.join(cluster_data['top_entities'][:10])}

Queries and Responses:
{queries_summary}
"""
            
            cluster_info = ClusterInfo(
                communityId=cluster_id,
                summary=cluster_data["summary"],
                related_queries=[q.query_text for q in cluster_data["queries"][:5]],
                entity_count=cluster_data["entity_count"],
                top_entities=cluster_data["top_entities"][:10],
                similarity_score=1.0  # Perfect match for direct cluster query
            )
            
            return QueryResponse(
                answer=answer,
                clusters=[cluster_info],
                query_embedding_dim=0
            )
            
        elif intent["type"] == "content_search":
            # Search clusters by content
            with driver.session() as session:
                search_results = session.read_transaction(search_clusters_by_content, user_query, top_k)
            
            if not search_results:
                return QueryResponse(
                    answer="No clusters found containing the specified content.",
                    clusters=[],
                    query_embedding_dim=0
                )
            
            # Get detailed info for matching clusters
            clusters = []
            context_parts = []
            
            with driver.session() as session:
                for result in search_results:
                    cluster_id = result["communityId"]
                    cluster_info = session.read_transaction(fetch_cluster_info, cluster_id)
                    cluster_info.similarity_score = 1.0  # High relevance for content match
                    clusters.append(cluster_info)
                    
                    context_parts.append(f"Cluster {cluster_id}: {cluster_info.summary} (found {result['matching_queries']} matching queries)")
            
            combined_context = "\n\n".join(context_parts)
            answer = generate_answer_with_groq(user_query, combined_context)
            
            return QueryResponse(
                answer=answer,
                clusters=clusters,
                query_embedding_dim=0
            )
        
        else:
            # Default semantic search (your existing logic)
            query_embedding = embedding_model.encode(user_query).tolist()
            search_results = vector_store.search(query_embedding, k=top_k)
            
            if not search_results:
                raise HTTPException(status_code=404, detail="No relevant clusters found. Try indexing communities first.")

            clusters = []
            with driver.session() as session:
                for community_id_str, similarity_score in search_results:
                    community_id = int(community_id_str.replace("community_", ""))
                    cluster_info = session.read_transaction(fetch_cluster_info, community_id)
                    cluster_info.similarity_score = similarity_score
                    clusters.append(cluster_info)

            valid_clusters = [c for c in clusters if c.summary]
            if not valid_clusters:
                return QueryResponse(
                    answer="No relevant information found in the knowledge graph.",
                    clusters=clusters,
                    query_embedding_dim=len(query_embedding)
                )

            combined_context = "\n\n".join(
                f"Cluster {c.communityId} (relevance: {c.similarity_score:.3f}): {c.summary}" 
                for c in valid_clusters
            )

            answer = generate_answer_with_groq(user_query, combined_context)

            return QueryResponse(
                answer=answer,
                clusters=clusters,
                query_embedding_dim=len(query_embedding)
            )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/index", response_model=IndexResponse)
def index_communities(request: IndexRequest):
    """Index specific community or all communities into vector store"""
    try:
        with driver.session() as session:
            if request.community_id == -1:
                # Index all communities
                community_ids = session.execute_read(get_all_communities)
            else:
                community_ids = [request.community_id]
            
            indexed_communities = []
            
            for community_id in community_ids:
                community_id_str = f"community_{community_id}"
                
                # Skip if already indexed and not forcing reindex
                if not request.force_reindex and community_id_str in vector_store.community_ids:
                    continue
                
                summary = session.execute_read(get_community_summary, community_id)
                
                if summary:
                    # Generate embedding using free model
                    summary_embedding = embedding_model.encode(summary).tolist()
                    
                    # Add to vector store
                    success = vector_store.add_embeddings(
                        community_id=community_id_str,
                        embedding=summary_embedding,
                        metadata={"community_id": community_id, "summary": summary}
                    )
                    
                    if success:
                        indexed_communities.append(community_id)

        return IndexResponse(
            message=f"Successfully indexed {len(indexed_communities)} communities",
            indexed_communities=indexed_communities
        )

    except Exception as e:
        logger.error(f"Error indexing communities: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

@app.get("/communities")
def list_communities():
    """List all communities in the knowledge graph"""
    try:
        with driver.session() as session:
            community_ids = session.read_transaction(get_all_communities)
            
        vector_stats = vector_store.get_stats()
        
        return {
            "total_communities": len(community_ids),
            "community_ids": community_ids,
            "indexed_communities": vector_stats["total_communities"],
            "vector_store_stats": vector_stats
        }
    except Exception as e:
        logger.error(f"Error listing communities: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing communities: {str(e)}")

@app.get("/communities/{community_id}")
def get_community_details(community_id: int):
    """Get details for a specific community"""
    try:
        with driver.session() as session:
            cluster_info = session.read_transaction(fetch_cluster_info, community_id)
            
        if not cluster_info.summary and cluster_info.entity_count == 0:
            raise HTTPException(status_code=404, detail=f"Community {community_id} not found")
            
        return cluster_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching community details: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching community: {str(e)}")

@app.get("/clusters/{cluster_id}/details", response_model=ClusterDetailResponse)
def get_cluster_queries_and_responses(cluster_id: int):
    """Get detailed information including all queries and responses for a specific cluster"""
    try:
        with driver.session() as session:
            cluster_data = session.read_transaction(fetch_cluster_queries_and_responses, cluster_id)
            
        if cluster_data["entity_count"] == 0:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
            
        return ClusterDetailResponse(
            cluster_id=cluster_id,
            summary=cluster_data["summary"],
            entity_count=cluster_data["entity_count"],
            top_entities=cluster_data["top_entities"],
            queries=cluster_data["queries"],
            total_queries=cluster_data["total_queries"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching cluster details: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching cluster details: {str(e)}")

@app.get("/search/clusters")
def search_clusters_by_content_endpoint(q: str, limit: int = 10):
    """Search for clusters that contain specific content in their queries/responses"""
    try:
        with driver.session() as session:
            results = session.read_transaction(search_clusters_by_content, q, limit)
            
        return {
            "search_term": q,
            "total_results": len(results),
            "clusters": results
        }
        
    except Exception as e:
        logger.error(f"Error searching clusters: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching clusters: {str(e)}")

@app.delete("/index")
def clear_index():
    """Clear the vector store index"""
    try:
        vector_store._initialize_empty_index()
        vector_store._save_index()
        return {"message": "Vector store index cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing index: {str(e)}")

@app.get("/vector-stats")
def get_vector_stats():
    """Get vector store statistics"""
    return vector_store.get_stats()

# Cleanup on shutdown
@app.on_event("shutdown")
def shutdown_event():
    driver.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)