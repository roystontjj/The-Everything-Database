import streamlit as st
import google.generativeai as genai
import time
from typing import List, Dict, Any
import pandas as pd
import traceback

def initialize_gemini_api():
    """Initialize the Gemini API client"""
    try:
        # Try to get API key from Streamlit secrets
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            # If not in secrets, get from session state
            api_key = st.session_state.gemini_api_key
            
        genai.configure(api_key=api_key)
        embedding_model = "models/embedding-001"
        return embedding_model
    except Exception as e:
        error_msg = f"Failed to initialize Gemini API: {str(e)}"
        st.error(error_msg)
        return None

def generate_embedding(text: str, model: str):
    """Generate embedding using Gemini API"""
    try:
        if not text or text.strip() == "":
            return None
        
        embedding = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_query" # Use retrieval_query for queries, retrieval_document for documents
        )
        
        # Return the embedding values as a list
        vector = embedding["embedding"]
        return vector
    except Exception as e:
        error_msg = f"Error generating embedding: {str(e)}"
        st.error(error_msg)
        time.sleep(1)  # Add a small delay in case of rate limiting
        return None

def vector_search(supabase, query_text: str, top_k: int = 5, threshold: float = 0.7):
    """
    Perform vector search across multiple tables
    
    Args:
        supabase: Supabase client instance
        query_text: The search query
        top_k: Number of results to return
        threshold: Similarity threshold
        
    Returns:
        List of search results with table, similarity score, and content
    """
    try:
        # Generate embedding for the query
        embedding_model = initialize_gemini_api()
        query_embedding = generate_embedding(query_text, embedding_model)
        
        if not query_embedding:
            st.warning("Could not generate embedding for the query")
            return []
        
        # Tables to search
        tables = [
            "charities",
            "charity_news",
            "charity_people",
            "charity_impact_areas",
            "charity_highlights",
            "charity_financials"
            # Exclude more complex tables until they're working properly
            # "charity_causes",
            # "charity_social_media",
            # "causes"
        ]
        
        # For debugging
        if "debug_mode" in st.session_state and st.session_state.debug_mode:
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
                
                if "debug_mode" in st.session_state and st.session_state.debug_mode:
                    st.write(f"Search results for {table_name}: {len(response.data or [])}")
                
                if response.data:
                    # Add table information to each result
                    for item in response.data:
                        # Retrieve additional information for certain tables
                        additional_info = {}
                        
                        # For charity results, get the charity name
                        if table_name == "charities" and "id" in item:
                            try:
                                charity_info = supabase.table("charities").select("name").eq("charity_id", item["id"]).execute()
                                if charity_info.data:
                                    additional_info["charity_name"] = charity_info.data[0]["name"]
                            except Exception as e:
                                if "debug_mode" in st.session_state and st.session_state.debug_mode:
                                    st.warning(f"Error getting charity name: {str(e)}")
                        
                        all_results.append({
                            'table': table_name,
                            'score': item['similarity'],
                            'content': item['content'],
                            'id': item['id'],
                            'additional_info': additional_info
                        })
            except Exception as table_error:
                if "debug_mode" in st.session_state and st.session_state.debug_mode:
                    st.warning(f"Error searching {table_name}: {str(table_error)}")
                continue
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Log results if in debug mode
        if "debug_mode" in st.session_state and st.session_state.debug_mode:
            st.write(f"Found {len(all_results)} relevant results across {len(tables)} tables")
            
        return all_results[:top_k]
        
    except Exception as e:
        st.error(f"Vector search error: {str(e)}")
        # Print stack trace for better debugging
        if "debug_mode" in st.session_state and st.session_state.debug_mode:
            st.error(traceback.format_exc())
        return []

def format_results_for_rag(results: List[Dict], query: str) -> str:
    """Format search results into context for RAG"""
    if not results:
        return "No relevant information found in the database for this query."
    
    formatted_context = "RETRIEVED INFORMATION:\n\n"
    
    for i, result in enumerate(results):
        table = result['table']
        content = result['content']
        score = result['score']
        additional_info = result.get('additional_info', {})
        
        # Format differently based on table type
        if table == "charities":
            charity_name = additional_info.get("charity_name", "Unknown Charity")
            formatted_context += f"CHARITY [{charity_name}]: {content}\n\n"
        elif table == "charity_highlights":
            formatted_context += f"HIGHLIGHT: {content}\n\n"
        elif table == "charity_impact_areas":
            formatted_context += f"IMPACT: {content}\n\n"
        elif table == "charity_news":
            formatted_context += f"NEWS: {content}\n\n"
        elif table == "charity_financials":
            formatted_context += f"FINANCIAL: {content}\n\n"
        elif table == "charity_causes":
            formatted_context += f"CAUSE RELATIONSHIP: {content}\n\n"
        elif table == "charity_people":
            formatted_context += f"PERSON: {content}\n\n"
        elif table == "causes":
            formatted_context += f"CAUSE: {content}\n\n"
        elif table == "charity_social_media":
            formatted_context += f"SOCIAL MEDIA: {content}\n\n"
        else:
            formatted_context += f"[{table.upper()}]: {content}\n\n"
    
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