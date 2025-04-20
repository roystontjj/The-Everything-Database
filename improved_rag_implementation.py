import streamlit as st
import google.generativeai as genai
import time
import traceback
from typing import List, Dict, Any

def rag_query(user_query: str, supabase_client, gemini_model, 
              top_k: int = 5, similarity_threshold: float = 0.7, 
              debug_mode: bool = False):
    """
    Complete RAG pipeline: embedding query, retrieving context, generating answer
    
    Args:
        user_query: The user's question or query
        supabase_client: Initialized Supabase client
        gemini_model: Initialized Gemini model
        top_k: Number of results to retrieve
        similarity_threshold: Minimum similarity score (0-1)
        debug_mode: Whether to show debugging information
        
    Returns:
        dict: Contains the generated answer, retrieval info, and context
    """
    start_time = time.time()
    debug_info = {}
    
    try:
        # Step 1: Generate embedding for the user query
        query_embedding_time = time.time()
        embedding_model = initialize_gemini_api()
        query_embedding = generate_embedding(
            user_query, 
            embedding_model, 
            task_type="retrieval_query"
        )
        
        if not query_embedding:
            return {
                "answer": "I'm unable to process your query at the moment. Please try again.",
                "error": "Failed to generate query embedding"
            }
            
        debug_info["embedding_time"] = time.time() - query_embedding_time
        debug_info["embedding_dimensions"] = len(query_embedding)
        
        # Step 2: Retrieve relevant documents based on vector similarity
        retrieval_time = time.time()
        search_results = perform_vector_search(
            supabase_client,
            query_embedding,
            user_query,
            top_k=top_k,
            threshold=similarity_threshold,
            debug_mode=debug_mode
        )
        
        debug_info["retrieval_time"] = time.time() - retrieval_time
        debug_info["num_results"] = len(search_results)
        
        # Step 3: Format retrieved documents into context
        context_time = time.time()
        formatted_context = format_results_for_rag(search_results, user_query)
        debug_info["context_time"] = time.time() - context_time
        
        # Step 4: Create the RAG prompt with query and context
        prompt_time = time.time()
        rag_prompt = create_rag_prompt(user_query, formatted_context)
        debug_info["prompt_time"] = time.time() - prompt_time
        
        # Step 5: Generate the final answer using Gemini
        generation_time = time.time()
        response = gemini_model.generate_content(rag_prompt)
        debug_info["generation_time"] = time.time() - generation_time
        
        # Step 6: Prepare the final response
        total_time = time.time() - start_time
        
        return {
            "answer": response.text,
            "context": formatted_context,
            "results": search_results,
            "total_time": total_time,
            "debug_info": debug_info
        }
        
    except Exception as e:
        error_message = f"RAG query error: {str(e)}"
        if debug_mode:
            error_message += f"\n{traceback.format_exc()}"
            
        return {
            "answer": "I encountered an error while processing your question. Please try again or rephrase your question.",
            "error": error_message
        }


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
        st.error(error_msg)
        time.sleep(1)  # Add a small delay in case of rate limiting
        return None


def perform_vector_search(supabase, query_embedding, query_text: str, 
                         top_k: int = 5, threshold: float = 0.7, 
                         debug_mode: bool = False):
    """
    Perform vector search across multiple tables using pre-computed embedding
    
    Args:
        supabase: Supabase client instance
        query_embedding: Pre-computed embedding vector for the query
        query_text: The original text query (for debugging)
        top_k: Number of results to return
        threshold: Similarity threshold
        debug_mode: Whether to show debugging information
        
    Returns:
        List of search results with table, similarity score, and content
    """
    try:
        if not query_embedding:
            st.warning("No query embedding provided for vector search")
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
        if debug_mode:
            st.write(f"Searching {len(tables)} tables with threshold {threshold}")
        
        # Perform vector search across tables
        all_results = []
        table_results = {}
        
        for table_name in tables:
            table_start_time = time.time()
            
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
                
                table_results[table_name] = {
                    "count": len(response.data or []),
                    "time": time.time() - table_start_time
                }
                
                if response.data:
                    # Add table information to each result
                    for item in response.data:
                        # Handle different table formats
                        if "id" not in item or "content" not in item or "similarity" not in item:
                            continue
                            
                        # Retrieve additional information for certain tables
                        additional_info = {}
                        
                        # For charity results, get the charity name
                        if table_name == "charities" and "id" in item:
                            try:
                                charity_info = supabase.table("charities").select("name").eq("charity_id", item["id"]).execute()
                                if charity_info.data:
                                    additional_info["charity_name"] = charity_info.data[0]["name"]
                            except Exception as e:
                                if debug_mode:
                                    st.warning(f"Error getting charity name: {str(e)}")
                        
                        # Add to results
                        all_results.append({
                            'table': table_name,
                            'score': item['similarity'],
                            'content': item['content'],
                            'id': item['id'],
                            'additional_info': additional_info
                        })
            except Exception as table_error:
                if debug_mode:
                    st.warning(f"Error searching {table_name}: {str(table_error)}")
                table_results[table_name] = {
                    "error": str(table_error),
                    "time": time.time() - table_start_time
                }
                continue
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Log results if in debug mode
        if debug_mode:
            st.write(f"Found {len(all_results)} relevant results across {len(tables)} tables")
            st.write("Table search results:", table_results)
            
        return all_results[:top_k]
        
    except Exception as e:
        st.error(f"Vector search error: {str(e)}")
        if debug_mode:
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