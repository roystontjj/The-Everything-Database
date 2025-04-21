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
    """
    try:
        if not query_embedding:
            st.warning("No query embedding provided for vector search")
            return []
        
        # First, specifically retrieve all charities with a lower threshold
        charity_threshold = max(0.5, threshold - 0.2)  # Lower threshold for charities
        charity_results = []
        
        try:
            # Get all charities first with lower threshold
            charity_response = supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': charity_threshold,
                    'match_count': 10,  # Get all 10 charities
                    'table_name': 'charities'
                }
            ).execute()
            
            if debug_mode:
                st.write(f"Found {len(charity_response.data or [])} charities with threshold {charity_threshold}")
                
            # Process charity results
            if charity_response.data:
                for item in charity_response.data:
                    if "id" not in item or "content" not in item or "similarity" not in item:
                        continue
                    
                    charity_info = {
                        'table': 'charities',
                        'score': item['similarity'],
                        'content': item['content'],
                        'id': item['id'],
                        'additional_info': {'charity_name': ''}
                    }
                    
                    # Get charity name
                    try:
                        name_info = supabase.table("charities").select("name").eq("charity_id", item["id"]).execute()
                        if name_info.data:
                            charity_info['additional_info']["charity_name"] = name_info.data[0]["name"]
                    except Exception as e:
                        if debug_mode:
                            st.warning(f"Error getting charity name: {str(e)}")
                    
                    charity_results.append(charity_info)
        except Exception as e:
            if debug_mode:
                st.warning(f"Error in charity search: {str(e)}")
        
        # Now search other tables with regular threshold
        other_tables = [
            "charity_news",
            "charity_people",
            "charity_impact_areas",
            "charity_highlights",
            "charity_financials",
            "charity_causes",
            "charity_social_media",
            "causes"
        ]
        
        other_results = []
        table_stats = {}
        
        for table_name in other_tables:
            table_start_time = time.time()
            
            try:
                response = supabase.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': threshold,
                        'match_count': top_k,
                        'table_name': table_name
                    }
                ).execute()
                
                table_stats[table_name] = {
                    "count": len(response.data or []),
                    "time": time.time() - table_start_time
                }
                
                if response.data:
                    for item in response.data:
                        if "id" not in item or "content" not in item or "similarity" not in item:
                            continue
                            
                        # Get the charity_id for this record if applicable
                        additional_info = {}
                        related_charity = None
                        
                        if table_name != "causes":
                            try:
                                # Different tables have different ID column names
                                id_column = table_name.replace("charity_", "") + "_id"
                                if table_name == "charity_causes":
                                    id_column = "charity_id"  # Special case
                                
                                # Get charity_id based on table structure
                                charity_id = None
                                
                                if table_name == "charity_causes":
                                    charity_id = item["id"]  # It's already the charity_id
                                else:
                                    record_info = supabase.table(table_name).select("charity_id").eq(id_column, item["id"]).execute()
                                    if record_info.data and len(record_info.data) > 0:
                                        charity_id = record_info.data[0]["charity_id"]
                                
                                # Get charity info
                                if charity_id:
                                    charity_info = supabase.table("charities").select("name").eq("charity_id", charity_id).execute()
                                    if charity_info.data and len(charity_info.data) > 0:
                                        additional_info["charity_name"] = charity_info.data[0]["name"]
                                        additional_info["charity_id"] = charity_id
                                        
                            except Exception as e:
                                if debug_mode:
                                    st.warning(f"Error getting related charity: {str(e)}")
                        
                        other_results.append({
                            'table': table_name,
                            'score': item['similarity'],
                            'content': item['content'],
                            'id': item['id'],
                            'additional_info': additional_info
                        })
            except Exception as e:
                if debug_mode:
                    st.warning(f"Error searching {table_name}: {str(e)}")
                table_stats[table_name] = {
                    "error": str(e),
                    "time": time.time() - table_start_time
                }
        
        # Combine results, ensuring charity results come first
        all_results = charity_results + other_results
        
        # Sort by similarity score (but keep all charities)
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Log stats if in debug mode
        if debug_mode:
            st.write(f"Found {len(charity_results)} charities and {len(other_results)} other results")
            st.write("Table search stats:", table_stats)
        
        return all_results[:max(top_k, len(charity_results))]
        
    except Exception as e:
        st.error(f"Vector search error: {str(e)}")
        if debug_mode:
            st.error(traceback.format_exc())
        return []


def format_results_for_rag(results: List[Dict], query: str) -> str:
    """Format search results into context for RAG, organizing by charity"""
    if not results:
        return "No relevant information found in the database for this query."
    
    # Group results by charity
    charity_data = {}
    other_results = []
    
    # First pass - collect all charity information
    for result in results:
        if result['table'] == "charities":
            charity_id = result['id']
            charity_name = result['additional_info'].get('charity_name', 'Unknown Charity')
            
            if charity_id not in charity_data:
                charity_data[charity_id] = {
                    'name': charity_name,
                    'description': result['content'],
                    'highlights': [],
                    'impact': [],
                    'news': [],
                    'financials': [],
                    'causes': [],
                    'people': [],
                    'social': []
                }
    
    # Second pass - add related information to charities
    for result in results:
        if result['table'] == "charities":
            continue  # Already processed
            
        charity_id = result['additional_info'].get('charity_id')
        
        # If we can link this to a charity
        if charity_id and charity_id in charity_data:
            if result['table'] == "charity_highlights":
                charity_data[charity_id]['highlights'].append(result['content'])
            elif result['table'] == "charity_impact_areas":
                charity_data[charity_id]['impact'].append(result['content'])
            elif result['table'] == "charity_news":
                charity_data[charity_id]['news'].append(result['content'])
            elif result['table'] == "charity_financials":
                charity_data[charity_id]['financials'].append(result['content'])
            elif result['table'] == "charity_causes":
                charity_data[charity_id]['causes'].append(result['content'])
            elif result['table'] == "charity_people":
                charity_data[charity_id]['people'].append(result['content'])
            elif result['table'] == "charity_social_media":
                charity_data[charity_id]['social'].append(result['content'])
        else:
            # Information not linked to a specific charity
            other_results.append(result)
    
    # Format the context
    formatted_context = "RETRIEVED INFORMATION:\n\n"
    
    # First, add organized charity information
    for charity_id, data in charity_data.items():
        formatted_context += f"CHARITY [{data['name']}]: {data['description']}\n"
        
        if data['highlights']:
            formatted_context += "  HIGHLIGHTS:\n"
            for highlight in data['highlights']:
                formatted_context += f"   - {highlight}\n"
                
        if data['impact']:
            formatted_context += "  IMPACT AREAS:\n"
            for impact in data['impact']:
                formatted_context += f"   - {impact}\n"
                
        if data['news']:
            formatted_context += "  NEWS:\n"
            for news in data['news']:
                formatted_context += f"   - {news}\n"
                
        if data['financials']:
            formatted_context += "  FINANCIALS:\n"
            for financial in data['financials']:
                formatted_context += f"   - {financial}\n"
                
        if data['causes']:
            formatted_context += "  CAUSES:\n"
            for cause in data['causes']:
                formatted_context += f"   - {cause}\n"
                
        if data['people']:
            formatted_context += "  PEOPLE:\n"
            for person in data['people']:
                formatted_context += f"   - {person}\n"
                
        if data['social']:
            formatted_context += "  SOCIAL MEDIA:\n"
            for social in data['social']:
                formatted_context += f"   - {social}\n"
                
        formatted_context += "\n"
    
    # Then add any other results
    for result in other_results:
        table = result['table']
        content = result['content']
        
        if table == "causes":
            formatted_context += f"CAUSE: {content}\n\n"
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