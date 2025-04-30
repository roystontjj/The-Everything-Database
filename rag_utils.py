import streamlit as st
import google.generativeai as genai
import time
from typing import List, Dict, Any
import pandas as pd
import traceback
from config import get_config

def initialize_gemini_api():
    """Initialize the Gemini API client"""
    try:
        # Get API key from session state or config
        api_key = st.session_state.get("gemini_api_key")
        
        # If not in session state, try from config
        if not api_key:
            config = get_config()
            api_key = config["gemini_api_key"]
            if api_key:
                st.session_state.gemini_api_key = api_key
        
        if not api_key:
            st.warning("Gemini API key not found. Please configure it in settings.")
            return None
            
        genai.configure(api_key=api_key)
        embedding_model = "models/embedding-001"
        return embedding_model
    except Exception as e:
        error_msg = f"Failed to initialize Gemini API: {str(e)}"
        if st.session_state.get("debug_mode", False):
            st.error(error_msg)
        return None

def generate_embedding(text: str, model: str, task_type: str = "retrieval_query"):
    """
    Generate embedding using Gemini API
    
    Args:
        text: Text to embed
        model: Embedding model to use
        task_type: Either "retrieval_query" (for user queries) or 
                  "retrieval_document" (for database documents)
    
    Returns:
        List of floats representing the embedding vector
    """
    try:
        if not text or text.strip() == "":
            return None
        
        embedding = genai.embed_content(
            model=model,
            content=text,
            task_type=task_type
        )
        
        # Return the embedding values as a list
        vector = embedding["embedding"]
        return vector
    except Exception as e:
        error_msg = f"Error generating embedding: {str(e)}"
        if st.session_state.get("debug_mode", False):
            st.error(error_msg)
        time.sleep(1)  # Add a small delay in case of rate limiting
        return None

def vector_search(supabase, query_text: str, top_k: int = 5, threshold: float = 0.7):
    """
    Perform vector search in the CharityFinder database
    
    Args:
        supabase: Supabase client instance
        query_text: The search query
        top_k: Number of results to return
        threshold: Similarity threshold
        
    Returns:
        List of search results with similarity score, and content
    """
    try:
        # Generate embedding for the query
        embedding_model = initialize_gemini_api()
        query_embedding = generate_embedding(query_text, embedding_model, task_type="retrieval_query")
        
        if not query_embedding:
            st.warning("Could not generate embedding for the query")
            return []
        
        # Tables to search (only charities in this version)
        tables = ["charities"]
        
        # For debugging
        if st.session_state.get("debug_mode", False):
            st.write(f"Generated embedding with {len(query_embedding)} dimensions")
            st.write(f"Searching {len(tables)} tables with threshold {threshold}")
        
        # Perform vector search across tables
        all_results = []
        
        for table_name in tables:
            try:
                # Use RPC function for vector search
                response = supabase.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': threshold,
                        'match_count': top_k,
                        'table_name': table_name
                    }
                ).execute()
                
                if st.session_state.get("debug_mode", False):
                    st.write(f"Search results for {table_name}: {len(response.data or [])}")
                
                if response.data:
                    # Add table information to each result
                    for item in response.data:
                        # Handle different table formats
                        if "id" not in item or "content" not in item or "similarity" not in item:
                            continue
                        
                        # Create result record
                        result = {
                            'table': table_name,
                            'score': item['similarity'],
                            'content': item['content'],
                            'id': item['id'],
                            'additional_info': {}
                        }
                        
                        # Retrieve charity name as additional information
                        try:
                            charity_info = supabase.table("charities").select("name, type, sector, classification").eq("id", item["id"]).execute()
                            if charity_info.data and len(charity_info.data) > 0:
                                result['additional_info']['name'] = charity_info.data[0].get("name", "Unknown")
                                result['additional_info']['type'] = charity_info.data[0].get("type", "Unknown")
                                result['additional_info']['sector'] = charity_info.data[0].get("sector", "Unknown")
                                result['additional_info']['classification'] = charity_info.data[0].get("classification", "Unknown")
                        except Exception as e:
                            if st.session_state.get("debug_mode", False):
                                st.warning(f"Error retrieving additional info: {str(e)}")
                        
                        all_results.append(result)
            except Exception as table_error:
                if st.session_state.get("debug_mode", False):
                    st.warning(f"Error searching {table_name}: {str(table_error)}")
                continue
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Log results if in debug mode
        if st.session_state.get("debug_mode", False):
            st.write(f"Found {len(all_results)} relevant results")
            
        return all_results[:top_k]
        
    except Exception as e:
        if st.session_state.get("debug_mode", False):
            st.error(f"Vector search error: {str(e)}")
            st.error(traceback.format_exc())
        return []

def format_results_for_rag(results: List[Dict], query: str) -> str:
    """Format search results into context for RAG"""
    if not results:
        return "No relevant information found in the database for this query."
    
    formatted_context = "RETRIEVED INFORMATION:\n\n"
    
    # Keep track of displayed charities to avoid duplication
    displayed_charities = set()
    
    for i, result in enumerate(results):
        charity_id = result['id']
        name = result['additional_info'].get('name', 'Unknown Charity')
        charity_type = result['additional_info'].get('type', 'Unknown Type')
        sector = result['additional_info'].get('sector', 'Unknown Sector')
        classification = result['additional_info'].get('classification', 'Unknown Classification')
        content = result['content']
        score = result['score']
        
        # Skip duplicates
        if charity_id in displayed_charities:
            continue
            
        displayed_charities.add(charity_id)
        
        # Format charity information
        formatted_context += f"CHARITY: {name}\n"
        formatted_context += f"TYPE: {charity_type}\n"
        formatted_context += f"SECTOR: {sector}\n"
        formatted_context += f"CLASSIFICATION: {classification}\n" 
        formatted_context += f"ACTIVITIES: {content}\n\n"
    
    return formatted_context

def create_rag_prompt(query: str, context: str) -> str:
    """Create a prompt for the LLM using RAG context"""
    return f"""
    You are a helpful charity information assistant. Answer the user's question based on the retrieved context below.
    If the question cannot be answered based on the context provided, say so, but try to be helpful using general 
    knowledge about charities.
    
    CONTEXT:
    {context}
    
    USER QUESTION: {query}
    
    Your answer should be comprehensive and directly address the user's question.
    Include specific details from the context when available.
    Format your answer nicely with appropriate paragraphs and sections when needed.
    """