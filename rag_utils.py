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
    Improved vector search function that properly handles joining between tables
    
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
        query_embedding = generate_embedding(query_text, embedding_model, task_type="retrieval_query")
        
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
            "charity_financials",
            "charity_causes",
            "charity_social_media",
            "causes"
        ]
        
        # For debugging
        if st.session_state.debug_mode:
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
                
                if st.session_state.debug_mode:
                    st.write(f"Search results for {table_name}: {len(response.data or [])}")
                
                if response.data:
                    # Add table information to each result
                    for item in response.data:
                        # Handle different table formats
                        if "id" not in item or "content" not in item or "similarity" not in item:
                            continue
                            
                        # Retrieve additional information for certain tables
                        additional_info = {}
                        
                        # IMPROVED: Special handling for charity_highlights table
                        if table_name == "charity_highlights":
                            try:
                                # Get the charity_id from the highlight
                                highlight_id = item["id"]
                                highlight_info = supabase.table("charity_highlights").select("charity_id").eq("highlight_id", highlight_id).execute()
                                
                                if highlight_info.data and len(highlight_info.data) > 0:
                                    charity_id = highlight_info.data[0]["charity_id"]
                                    
                                    # Get the charity details
                                    charity_info = supabase.table("charities").select("charity_id,name,description").eq("charity_id", charity_id).execute()
                                    
                                    if charity_info.data and len(charity_info.data) > 0:
                                        # Include full charity information
                                        additional_info["charity_id"] = charity_id
                                        additional_info["charity_name"] = charity_info.data[0]["name"]
                                        additional_info["charity_description"] = charity_info.data[0]["description"]
                                        
                                        # This is important: we modify the content to include charity info
                                        item["content"] = f"Charity '{charity_info.data[0]['name']}' is a {item['content']}"
                            except Exception as e:
                                if st.session_state.debug_mode:
                                    st.warning(f"Error enriching highlight data: {str(e)}")
                        
                        # Regular handling for charities table
                        elif table_name == "charities" and "id" in item:
                            try:
                                charity_id = item["id"]
                                charity_info = supabase.table("charities").select("name").eq("charity_id", charity_id).execute()
                                if charity_info.data:
                                    additional_info["charity_name"] = charity_info.data[0]["name"]
                                    additional_info["charity_id"] = charity_id
                            except Exception as e:
                                if st.session_state.debug_mode:
                                    st.warning(f"Error getting charity name: {str(e)}")
                        
                        all_results.append({
                            'table': table_name,
                            'score': item['similarity'],
                            'content': item['content'],
                            'id': item['id'],
                            'additional_info': additional_info
                        })
            except Exception as table_error:
                if st.session_state.debug_mode:
                    st.warning(f"Error searching {table_name}: {str(table_error)}")
                continue
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Log results if in debug mode
        if st.session_state.debug_mode:
            st.write(f"Found {len(all_results)} relevant results across {len(tables)} tables")
            
        return all_results[:top_k]
        
    except Exception as e:
        if st.session_state.debug_mode:
            st.error(f"Vector search error: {str(e)}")
            st.error(traceback.format_exc())
        return []

def format_results_for_rag(results: List[Dict], query: str) -> str:
    """
    Enhanced formatter that provides better display of joined charity highlight information
    
    Args:
        results: List of search results with enriched data
        query: Original user query
        
    Returns:
        Formatted context string for the RAG prompt
    """
    if not results:
        return "No relevant information found in the database for this query."
    
    formatted_context = "RETRIEVED INFORMATION:\n\n"
    
    # Keep track of displayed charities to avoid duplication
    displayed_charities = set()
    
    for i, result in enumerate(results):
        table = result['table']
        content = result['content']
        score = result['score']
        additional_info = result.get('additional_info', {})
        
        # Format differently based on table type
        if table == "charities":
            charity_name = additional_info.get("charity_name", "Unknown Charity")
            charity_id = additional_info.get("charity_id")
            
            if charity_id and charity_id in displayed_charities:
                # Skip if we've already shown this charity
                continue
                
            if charity_id:
                displayed_charities.add(charity_id)
                
            formatted_context += f"CHARITY [{charity_name}]: {content}\n\n"
            
        elif table == "charity_highlights":
            # Enhanced highlight display with charity information
            charity_name = additional_info.get("charity_name")
            charity_description = additional_info.get("charity_description")
            charity_id = additional_info.get("charity_id")
            
            if charity_id:
                # Only add the charity ID to the displayed set if we have full info
                if charity_id in displayed_charities:
                    formatted_context += f"HIGHLIGHT: {content}\n\n"
                else:
                    displayed_charities.add(charity_id)
                    # Include full charity information with the highlight
                    formatted_context += f"CHARITY [{charity_name}]: {charity_description}\n"
                    formatted_context += f"HIGHLIGHT: {content}\n\n"
            else:
                # Simple highlight without charity information
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