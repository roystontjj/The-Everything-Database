import streamlit as st
import google.generativeai as genai
import pandas as pd
from supabase import create_client
import requests
import json
import time
import traceback
from typing import List, Dict, Any

# Page config and styling
st.set_page_config(page_title="Gemini AI Charity RAG Assistant", layout="wide")

# Custom CSS for better UI with improved contrast
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .chat-container {
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .user-message {
        background-color: #2b3a55;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .assistant-message {
        background-color: #3e4a61;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .retrieved-context {
        background-color: #1e2b3e;
        color: #a0c0e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-family: monospace;
        font-size: 0.9em;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "Main Dashboard"
if "supabase_url" not in st.session_state:
    st.session_state.supabase_url = ""
if "supabase_key" not in st.session_state:
    st.session_state.supabase_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "batch_size" not in st.session_state:
    st.session_state.batch_size = 10
if "processing" not in st.session_state:
    st.session_state.processing = False
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "logs" not in st.session_state:
    st.session_state.logs = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Get API keys from Streamlit secrets
try:
    # IMPORTANT: Use secrets safely - API keys removed as requested
    # You will need to set these in Streamlit secrets or input them in the UI
    if st.secrets.get("GEMINI_API_KEY"):
        api_key = st.secrets["GEMINI_API_KEY"]
        st.session_state.gemini_api_key = api_key
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini model
        if "chat" not in st.session_state and api_key:
            model = genai.GenerativeModel('gemini-1.5-flash')
            st.session_state.model = model
            st.session_state.chat = model.start_chat(history=[])
except Exception as e:
    # We'll handle this in the UI - don't show error here
    pass

# Initialize Supabase client
try:
    # IMPORTANT: Use secrets safely - API keys removed as requested
    # You will need to set these in Streamlit secrets or input them in the UI
    if st.secrets.get("SUPABASE_URL") and st.secrets.get("SUPABASE_KEY"):
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        st.session_state.supabase_url = supabase_url
        st.session_state.supabase_key = supabase_key
        
        supabase = create_client(supabase_url, supabase_key)
        st.session_state.supabase_client = supabase
except Exception as e:
    # We'll handle this in the UI - don't show error here
    pass

# Function to test the Supabase connection
def test_connection():
    try:
        # Just fetch a small amount of data to test the connection
        response = st.session_state.supabase_client.table('charities').select('*').limit(1).execute()
        st.success("✅ Successfully connected to Supabase!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to connect to Supabase API: {e}")
        return False

# Function to execute query via Supabase API
def execute_query(table_name, select_columns="*", filters=None, order_by=None, limit=None):
    try:
        # Start with a basic query
        query = st.session_state.supabase_client.table(table_name).select(select_columns)
        
        # Apply filters if provided
        if filters:
            for column, value in filters.items():
                query = query.eq(column, value)
        
        # Apply order by if provided
        if order_by:
            query = query.order(order_by)
            
        # Apply limit if provided
        if limit:
            query = query.limit(limit)
            
        # Execute the query
        response = query.execute()
        
        # Convert to pandas DataFrame
        if response.data:
            df = pd.DataFrame(response.data)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Query execution error: {e}")
        return None

# Main Dashboard Page
def show_main_dashboard():
    st.title("Charity Data Dashboard")
    st.markdown("Welcome to the Charity Data Dashboard with RAG (Retrieval Augmented Generation).")
    
    # Display some basic statistics or information
    st.subheader("Database Overview")
    
    # If we have a Supabase connection, show basic stats
    if "supabase_client" in st.session_state:
        try:
            charities = execute_query("charities")
            news = execute_query("charity_news")
            people = execute_query("charity_people")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Charities", len(charities) if charities is not None and not charities.empty else 0)
            with col2:
                st.metric("News Items", len(news) if news is not None and not news.empty else 0)
            with col3:
                st.metric("People", len(people) if people is not None and not people.empty else 0)
            with col4:
                # Count records with missing embeddings
                if charities is not None and not charities.empty:
                    missing_embeddings = charities['embedding'].isna().sum()
                    st.metric("Missing Embeddings", missing_embeddings)
                
            # Display RAG status
            st.subheader("RAG System Status")
            
            # Check if the match_documents function exists
            try:
                # Try a simple vector search as a test
                test_query = "test query"
                embedding_model = "models/embedding-001"
                query_embedding = genai.embed_content(
                    model=embedding_model,
                    content=test_query,
                    task_type="retrieval_query"
                )["embedding"]
                
                response = st.session_state.supabase_client.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': 0.5,
                        'match_count': 1,
                        'table_name': 'charities'
                    }
                ).execute()
                
                st.success("✅ RAG system is operational! Vector search is working.")
            except Exception as e:
                st.error(f"❌ RAG system is not fully set up: {str(e)}")
                st.warning("You may need to create the 'match_documents' SQL function in your Supabase database.")
                
                with st.expander("Show SQL Function Code"):
                    st.code("""
-- Function for matching documents based on embedding similarity
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector,
  match_threshold float,
  match_count int,
  table_name text
)
RETURNS TABLE (
  id integer,
  content text,
  similarity float
)
LANGUAGE plpgsql
AS $$
DECLARE
  query_text TEXT;
  id_column TEXT;
  text_column TEXT;
  embedding_column TEXT;
BEGIN
  -- Set the appropriate columns based on the table name
  CASE table_name
    WHEN 'charities' THEN
      id_column := 'charity_id';
      text_column := 'description';
      embedding_column := 'embedding';
    WHEN 'charity_news' THEN
      id_column := 'news_id';
      text_column := 'content';
      embedding_column := 'embedding';
    -- Add more cases for other tables
    ELSE
      RAISE EXCEPTION 'Unsupported table: %', table_name;
  END CASE;

  -- Construct and execute dynamic SQL query
  query_text := format('
    SELECT %I as id, %I as content, 
           1 - (%I <=> $1) as similarity
    FROM %I
    WHERE %I IS NOT NULL AND 1 - (%I <=> $1) > $2
    ORDER BY similarity DESC
    LIMIT $3
  ', id_column, text_column, embedding_column, table_name, embedding_column, embedding_column);
  
  RETURN QUERY EXECUTE query_text USING query_embedding, match_threshold, match_count;
END;
$$;
                    """, language="sql")
                
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")
    else:
        st.info("Connect to your Supabase database to see statistics.")

# Data Viewer Page
def show_data_viewer():
    st.title("Data Viewer")
    
    if "supabase_client" not in st.session_state:
        st.warning("Please connect to your Supabase database first.")
        return
    
    # Select table to view
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
    
    selected_table = st.selectbox("Select table to view", tables)
    
    # Fetch and display data
    data = execute_query(selected_table)
    
    if data is not None and not data.empty:
        st.subheader(f"{selected_table} data")
        st.dataframe(data)
        
        # Download button for CSV
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"{selected_table}.csv",
            mime="text/csv"
        )
    else:
        st.info(f"No data found in {selected_table} table")

# IMPROVED RAG IMPLEMENTATION
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
        if st.session_state.debug_mode:
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
        if st.session_state.debug_mode:
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
                        
                        # For charity results, get the charity name
                        if table_name == "charities" and "id" in item:
                            try:
                                charity_info = supabase.table("charities").select("name").eq("charity_id", item["id"]).execute()
                                if charity_info.data:
                                    additional_info["charity_name"] = charity_info.data[0]["name"]
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
        # Step 1: Retrieve relevant documents based on vector similarity
        retrieval_time = time.time()
        search_results = vector_search(
            supabase_client,
            user_query,
            top_k=top_k,
            threshold=similarity_threshold
        )
        
        debug_info["retrieval_time"] = time.time() - retrieval_time
        debug_info["num_results"] = len(search_results)
        
        # Step 2: Format retrieved documents into context
        context_time = time.time()
        formatted_context = format_results_for_rag(search_results, user_query)
        debug_info["context_time"] = time.time() - context_time
        
        # Step 3: Create the RAG prompt with query and context
        prompt_time = time.time()
        rag_prompt = create_rag_prompt(user_query, formatted_context)
        debug_info["prompt_time"] = time.time() - prompt_time
        
        # Step 4: Generate the final answer using Gemini
        generation_time = time.time()
        response = gemini_model.generate_content(rag_prompt)
        debug_info["generation_time"] = time.time() - generation_time
        
        # Step 5: Prepare the final response
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

# IMPROVED RAG CHAT INTERFACE
def show_rag_chat_interface():
    """
    Enhanced RAG chat interface with improved query processing and debugging
    """
    st.title("💬 Charity RAG Assistant")
    st.markdown("Ask questions about charities using Retrieval Augmented Generation!")
    
    # Check if API is configured
    if not st.session_state.get("gemini_api_key") or not st.session_state.get("supabase_client"):
        st.warning("⚠️ Please configure your API keys in the Settings page before using the chat.")
        return

    # Display chat history
    st.subheader("Conversation")
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'><strong>You:</strong><br>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong><br>{message['content']}</div>", unsafe_allow_html=True)
                # Show retrieved context if available and debug mode is on
                if "context" in message and st.session_state.debug_mode:
                    with st.expander("View retrieved context"):
                        st.markdown(f"<div class='retrieved-context'>{message['context']}</div>", unsafe_allow_html=True)
                    
                    if "debug_info" in message and st.session_state.debug_mode:
                        with st.expander("View RAG debug information"):
                            st.json(message["debug_info"])

    # User input section
    st.subheader("Your Question")
    with st.form(key="prompt_form", clear_on_submit=True):
        user_prompt = st.text_area("Type your message here:", height=100, key="prompt_input")
        
        # RAG settings
        col1, col2, col3 = st.columns(3)
        with col1:
            use_rag = st.checkbox("Use RAG", value=True, 
                               help="Enable Retrieval Augmented Generation")
        with col2:
            num_results = st.slider("Results to retrieve", min_value=1, max_value=10, value=3)
        with col3:
            similarity_threshold = st.slider("Similarity threshold", min_value=0.5, max_value=0.95, value=0.7, step=0.05)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.form_submit_button("Send", use_container_width=True)
        with col2:
            if st.form_submit_button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if "model" in st.session_state:
                    st.session_state.chat = st.session_state.model.start_chat(history=[])
                st.rerun()

    # Process the user input when the form is submitted
    if submit_button and user_prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Show a spinner while processing
        with st.spinner("Processing your question..."):
            try:
                if use_rag and "supabase_client" in st.session_state:
                    # Use the enhanced RAG query function
                    rag_response = rag_query(
                        user_prompt,
                        st.session_state.supabase_client,
                        st.session_state.model,
                        top_k=num_results,
                        similarity_threshold=similarity_threshold,
                        debug_mode=st.session_state.debug_mode
                    )
                    
                    # Extract the answer and metadata
                    answer = rag_response.get("answer", "I couldn't process your question.")
                    context_html = rag_response.get("context", "").replace("\n", "<br>")
                    
                    # Add assistant response to chat history with context
                    response_message = {
                        "role": "assistant", 
                        "content": answer,
                        "context": context_html,
                    }
                    
                    # Add debug info if available
                    if "debug_info" in rag_response:
                        response_message["debug_info"] = rag_response["debug_info"]
                        
                    # Add total time if available
                    if "total_time" in rag_response:
                        response_message["retrieval_time"] = f"{rag_response['total_time']:.2f} seconds"
                        
                    st.session_state.messages.append(response_message)
                    
                else:
                    # Regular chat without RAG
                    response = st.session_state.chat.send_message(user_prompt)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
        
        # Rerun to update the UI with the new messages
        st.rerun()

    # Add advanced settings in an expander
    with st.expander("Advanced Settings"):
        st.checkbox("Show debugging information", value=st.session_state.debug_mode, key="debug_toggle")
        if st.button("Apply Settings"):
            st.session_state.debug_mode = st.session_state.get("debug_toggle", False)
            st.success("Settings applied!")
            
    # Explanation of RAG settings
    with st.expander("About RAG Settings"):
        st.markdown("""
        ### Retrieval Augmented Generation (RAG) Settings
        
        - **Use RAG**: Enable or disable the retrieval system. When enabled, the assistant will search for relevant information in the charity database before answering.
        
        - **Results to retrieve**: Control how many documents to retrieve from the database. More results provide more context but might include less relevant information.
        
        - **Similarity threshold**: Set the minimum similarity score (0.5-0.95) required for retrieved content. Higher values ensure more relevant results but might return fewer documents.
        
        When debug mode is enabled, you can view the retrieved context and detailed performance metrics for each query.
        """)

# IMPROVED EMBEDDING GENERATOR
def show_embedding_generator():
    """
    Shows the embedding generator interface
    """
    st.title("Database Embeddings Generator")
    
    # Check if API keys and client are set up
    if not st.session_state.get("gemini_api_key") or not st.session_state.get("supabase_client"):
        st.warning("Please configure your API keys in Settings before proceeding.")
        return
    
    # Initialize embedding model
    embedding_model = initialize_gemini_api()
    if not embedding_model:
        st.error("Failed to initialize embedding model. Please check your API key.")
        return
    
    # Define tables and their configurations
    TABLE_CONFIGS = {
        "charities": {
            "id_column": "charity_id",
            "text_column": "description", 
            "embedding_column": "embedding"
        },
        "charity_news": {
            "id_column": "news_id",
            "text_column": "content",
            "embedding_column": "embedding"
        },
        "charity_people": {
            "id_column": "person_id",
            "text_column": "name",
            "embedding_column": "embedding"
        },
        "charity_impact_areas": {
            "id_column": "impact_id",
            "text_column": "impact_description",
            "embedding_column": "embedding"
        },
        "charity_highlights": {
            "id_column": "highlight_id",
            "text_column": "highlight_text",
            "embedding_column": "embedding"
        },
        "charity_financials": {
            "id_column": "financial_id",
            "text_column": "other_financial_metrics",
            "embedding_column": "embedding"
        },
        "charity_causes": {
            "id_column": "charity_id",
            "text_column": None,  # Special case, will be handled differently
            "embedding_column": "embedding"
        },
        "charity_social_media": {
            "id_column": "social_id",
            "text_column": "platform",
            "embedding_column": "embedding"
        },
        "causes": {
            "id_column": "cause_id",
            "text_column": "description",
            "embedding_column": "embedding"
        }
    }
    
    # Operation settings
    st.subheader("Embedding Generation Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_tables = st.multiselect(
            "Select tables to process",
            options=list(TABLE_CONFIGS.keys()),
            default=list(TABLE_CONFIGS.keys())
        )
        
    with col2:
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=50,
            value=st.session_state.batch_size,
            help="Number of records to process in one batch"
        )
        st.session_state.batch_size = batch_size
    
    process_btn = st.button("Generate Embeddings", type="primary", use_container_width=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Display processing logs
    st.subheader("Processing Logs")
    logs_placeholder = st.empty()
    
    # Display table statistics
    st.subheader("Table Statistics")
    stats_placeholder = st.container()
    
    # Process tables when button is clicked
    if process_btn:
        st.session_state.processing = True
        st.session_state.progress = 0
        st.session_state.logs = []
        
        # Run the processing
        try:
            with st.spinner("Processing tables..."):
                process_tables(
                    selected_tables, 
                    TABLE_CONFIGS, 
                    batch_size,
                    progress_bar,
                    logs_placeholder,
                    stats_placeholder
                )
        except Exception as e:
            log_message(f"Error: {str(e)}", logs_placeholder)
            if st.session_state.get("debug_mode", False):
                st.error(traceback.format_exc())
        finally:
            st.session_state.processing = False
            
    # Always update the table statistics
    update_table_statistics(TABLE_CONFIGS, stats_placeholder)


def process_tables(tables, table_configs, batch_size, progress_bar, logs_placeholder, stats_placeholder):
    """Process multiple tables and generate embeddings"""
    total_tables = len(tables)
    table_counter = 0
    
    for table_name in tables:
        table_counter += 1
        config = table_configs[table_name]
        
        # Update progress
        progress = (table_counter - 1) / total_tables
        progress_bar.progress(progress)
        
        log_message(f"Processing table: {table_name}", logs_placeholder)
        
        # Special handling for charity_causes table
        if table_name == "charity_causes":
            process_charity_causes_table(table_name, config, batch_size, logs_placeholder)
        else:
            process_regular_table(table_name, config, batch_size, logs_placeholder)
            
        # Update statistics
        update_table_statistics(table_configs, stats_placeholder)
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    log_message("Embedding generation completed!", logs_placeholder)


def process_regular_table(table_name, config, batch_size, logs_placeholder):
    """Process a regular table to generate embeddings"""
    try:
        supabase = st.session_state.supabase_client
        embedding_model = initialize_gemini_api()
        
        # Get column configuration
        id_column = config["id_column"]
        text_column = config["text_column"]
        embedding_column = config["embedding_column"]
        
        # Fetch records with NULL embeddings
        log_message(f"Fetching records with NULL embeddings from {table_name}", logs_placeholder)
        
        response = supabase.table(table_name).select(f"{id_column}, {text_column}").is_(embedding_column, "null").execute()
        null_records = response.data
        
        log_message(f"Found {len(null_records)} records with NULL embeddings in {table_name}", logs_placeholder)
        
        # Fetch records with missing embeddings
        missing_records = []
        try:
            # Some databases might not distinguish between NULL and missing columns
            # This might fail if the column doesn't exist or if the SQL query syntax is different
            response = supabase.table(table_name).select(f"{id_column}, {text_column}").not_.is_(embedding_column, "null").eq(embedding_column, "").execute()
            missing_records = response.data
            log_message(f"Found {len(missing_records)} records with missing embeddings in {table_name}", logs_placeholder)
        except Exception as e:
            # If this fails, just continue with NULL records
            log_message(f"Skipping check for empty embeddings: {str(e)}", logs_placeholder)
        
        # Combine records to process
        records_to_process = null_records + missing_records
        
        if not records_to_process:
            log_message(f"No records found with missing embeddings in {table_name}", logs_placeholder)
            return
        
        # Process records in batches
        total_batches = (len(records_to_process) + batch_size - 1) // batch_size
        
        for batch_index in range(total_batches):
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, len(records_to_process))
            batch = records_to_process[start_idx:end_idx]
            
            log_message(f"Processing batch {batch_index + 1}/{total_batches} ({len(batch)} records)", logs_placeholder)
            
            for record in batch:
                record_id = record[id_column]
                text = record.get(text_column, "")
                
                # Skip records with empty text
                if not text or text.strip() == "":
                    log_message(f"Skipping {table_name} record with empty text", logs_placeholder)
                    continue
                    
                # Generate embedding
                try:
                    embedding = generate_embedding(text, embedding_model, "retrieval_document")
                    
                    if embedding:
                        # Update record with embedding
                        try:
                            # Use RPC to update embedding
                            response = supabase.rpc(
                                'update_embedding_vector',
                                {
                                    'p_table_name': table_name,
                                    'p_id_column': id_column,
                                    'p_id_value': record_id,
                                    'p_embedding_column': embedding_column,
                                    'p_embedding': embedding
                                }
                            ).execute()
                        except Exception as update_error:
                            # Fall back to direct update if RPC fails
                            log_message(f"RPC update failed, trying direct update: {str(update_error)}", logs_placeholder)
                            response = supabase.table(table_name).update({embedding_column: embedding}).eq(id_column, record_id).execute()
                            
                except Exception as e:
                    log_message(f"Error generating embedding for {table_name} record {record_id}: {str(e)}", logs_placeholder)
                    time.sleep(1)  # Add delay to avoid rate limits
            
            # Add a small delay between batches
            time.sleep(0.5)
            
    except Exception as e:
        log_message(f"Error processing table {table_name}: {str(e)}", logs_placeholder)
        if st.session_state.get("debug_mode", False):
            log_message(traceback.format_exc(), logs_placeholder)


def process_charity_causes_table(table_name, config, batch_size, logs_placeholder):
    """Special handling for charity_causes table which needs joined data"""
    try:
        supabase = st.session_state.supabase_client
        embedding_model = initialize_gemini_api()
        
        # Get column configuration
        id_column = config["id_column"]  # This is charity_id
        embedding_column = config["embedding_column"]
        
        # Fetch records with NULL embeddings
        log_message(f"Fetching records with NULL embeddings from {table_name}", logs_placeholder)
        
        # We need to join with charities and causes tables
        sql_query = f"""
        SELECT 
            cc.{id_column}, 
            cc.cause_id,
            c.name as charity_name,
            cs.name as cause_name
        FROM 
            {table_name} cc
        JOIN 
            charities c ON cc.charity_id = c.charity_id
        JOIN 
            causes cs ON cc.cause_id = cs.cause_id
        WHERE 
            cc.{embedding_column} IS NULL
        """
        
        try:
            response = supabase.query(sql_query).execute()
            records_to_process = response.data
        except Exception as e:
            # SQL query might fail, try a different approach
            log_message(f"SQL query failed, trying alternative approach: {str(e)}", logs_placeholder)
            
            # Get all records without embeddings
            response = supabase.table(table_name).select(f"{id_column}, cause_id").is_(embedding_column, "null").execute()
            base_records = response.data
            
            # For each record, get the names from other tables
            records_to_process = []
            for record in base_records:
                charity_id = record[id_column]
                cause_id = record["cause_id"]
                
                # Get charity name
                charity_response = supabase.table("charities").select("name").eq("charity_id", charity_id).execute()
                charity_name = charity_response.data[0]["name"] if charity_response.data else "Unknown Charity"
                
                # Get cause name
                cause_response = supabase.table("causes").select("name").eq("cause_id", cause_id).execute()
                cause_name = cause_response.data[0]["name"] if cause_response.data else "Unknown Cause"
                
                records_to_process.append({
                    id_column: charity_id,
                    "cause_id": cause_id,
                    "charity_name": charity_name,
                    "cause_name": cause_name
                })
        
        log_message(f"Found {len(records_to_process)} records with NULL embeddings in {table_name}", logs_placeholder)
        
        if not records_to_process:
            log_message(f"No records found with missing embeddings in {table_name}", logs_placeholder)
            return
        
        # Process records in batches
        total_batches = (len(records_to_process) + batch_size - 1) // batch_size
        
        for batch_index in range(total_batches):
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, len(records_to_process))
            batch = records_to_process[start_idx:end_idx]
            
            log_message(f"Processing batch {batch_index + 1}/{total_batches} ({len(batch)} records)", logs_placeholder)
            
            for record in batch:
                charity_id = record[id_column]
                cause_id = record["cause_id"]
                charity_name = record.get("charity_name", "Unknown Charity")
                cause_name = record.get("cause_name", "Unknown Cause")
                
                # Create text for embedding
                text = f'Charity "{charity_name}" supports cause "{cause_name}"'
                
                # Generate embedding
                try:
                    embedding = generate_embedding(text, embedding_model, "retrieval_document")
                    
                    if embedding:
                        # Update record with embedding
                        try:
                            # For charity_causes we need a compound WHERE clause
                            response = supabase.table(table_name).update(
                                {embedding_column: embedding}
                            ).eq(id_column, charity_id).eq("cause_id", cause_id).execute()
                            
                        except Exception as update_error:
                            log_message(f"Error updating embedding for {table_name} record {charity_id}-{cause_id}: {str(update_error)}", logs_placeholder)
                            
                except Exception as e:
                    log_message(f"Error generating embedding for {table_name} record {charity_id}-{cause_id}: {str(e)}", logs_placeholder)
                    time.sleep(1)  # Add delay to avoid rate limits
            
            # Add a small delay between batches
            time.sleep(0.5)
            
    except Exception as e:
        log_message(f"Error processing charity_causes table: {str(e)}", logs_placeholder)
        if st.session_state.get("debug_mode", False):
            log_message(traceback.format_exc(), logs_placeholder)


def update_table_statistics(table_configs, placeholder):
    """Update the table statistics display"""
    try:
        supabase = st.session_state.supabase_client
        
        with placeholder:
            stats = {}
            
            for table_name, config in table_configs.items():
                try:
                    # Get total count
                    total_response = supabase.table(table_name).select("*", count="exact").execute()
                    total_count = total_response.count
                    
                    # Get count of records with embeddings
                    embedding_column = config["embedding_column"]
                    with_embedding_response = supabase.table(table_name).select("*", count="exact").not_.is_(embedding_column, "null").execute()
                    with_embedding_count = with_embedding_response.count
                    
                    # Calculate missing count
                    missing_count = total_count - with_embedding_count
                    
                    # Calculate completion percentage
                    completion_percentage = (with_embedding_count / total_count * 100) if total_count > 0 else 0
                    
                    # Store the statistics
                    stats[table_name] = {
                        "total": total_count,
                        "missing": missing_count,
                        "completion": f"{completion_percentage:.1f}%"
                    }
                except Exception as e:
                    # If there's an error getting stats for this table, just continue
                    if st.session_state.get("debug_mode", False):
                        st.warning(f"Error getting stats for {table_name}: {str(e)}")
            
            # Display statistics
            for table_name, table_stats in stats.items():
                st.write(f"{table_name}: {table_stats['missing']} missing out of {table_stats['total']} ({table_stats['completion']} complete)")
    
    except Exception as e:
        st.error(f"Error updating statistics: {str(e)}")


def log_message(message, placeholder):
    """Add a log message and update the display"""
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    # Add to session state logs
    if "logs" not in st.session_state:
        st.session_state.logs = []
    
    st.session_state.logs.append(log_entry)
    
    # Keep only the last 100 log entries
    if len(st.session_state.logs) > 100:
        st.session_state.logs = st.session_state.logs[-100:]
    
    # Update the display
    with placeholder:
        st.code("\n".join(st.session_state.logs))


# Settings Page
def show_settings():
    st.title("Settings")
    
    st.subheader("Database Connection")
    
    # Supabase settings
    col1, col2 = st.columns(2)
    with col1:
        supabase_url = st.text_input("Supabase URL", value=st.session_state.supabase_url, key="settings_supabase_url")
    with col2:
        supabase_key = st.text_input("Supabase API Key", type="password", value=st.session_state.supabase_key, key="settings_supabase_key")
    
    # Gemini API settings    
    gemini_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key, key="settings_gemini_api_key")
    
    # Save settings button
    if st.button("Save Settings"):
        st.session_state.supabase_url = supabase_url
        st.session_state.supabase_key = supabase_key
        st.session_state.gemini_api_key = gemini_api_key
        
        # Update connections
        try:
            if supabase_url and supabase_key:
                supabase = create_client(supabase_url, supabase_key)
                st.session_state.supabase_client = supabase
                st.success("Supabase settings updated successfully!")
            
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                st.session_state.model = model
                st.session_state.chat = model.start_chat(history=[])
                st.success("Gemini API settings updated successfully!")
        except Exception as e:
            st.error(f"Error updating settings: {str(e)}")
    
    # Advanced Settings
    st.subheader("Advanced Settings")
    
    with st.expander("RAG Settings"):
        st.checkbox("Debug Mode", value=st.session_state.debug_mode, key="debug_mode_setting")
        if st.button("Save Advanced Settings"):
            st.session_state.debug_mode = st.session_state.get("debug_mode_setting", False)
            st.success("Advanced settings saved!")
    
    # API Testing
    st.subheader("API Testing")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Supabase Connection"):
            if "supabase_client" in st.session_state:
                test_connection()
            else:
                st.error("Please configure and save Supabase settings first.")
    
    with col2:
        if st.button("Test Vector Search"):
            if not st.session_state.get("supabase_client"):
                st.error("Please configure Supabase connection first")
            elif not st.session_state.get("gemini_api_key"):
                st.error("Please configure Gemini API key first")
            else:
                try:
                    # Test vector search with a simple query
                    test_results = vector_search(
                        st.session_state.supabase_client,
                        "Show me information about charity impact",
                        top_k=3,
                        threshold=0.6
                    )
                    
                    if test_results:
                        st.success(f"✅ Vector search successful! Found {len(test_results)} results.")
                        st.json(test_results)
                    else:
                        st.warning("Vector search returned no results. This might be normal if your database is empty or has no embeddings yet.")
                        
                except Exception as e:
                    st.error(f"Vector search test failed: {str(e)}")
                    
    # Test embedding generation
    st.subheader("Test Embedding Generation")
    
    test_text = st.text_input("Enter text to generate embedding:", value="Test charity information")
    if st.button("Generate Test Embedding"):
        try:
            if st.session_state.gemini_api_key:
                genai.configure(api_key=st.session_state.gemini_api_key)
                embedding_model = "models/embedding-001"
                
                # Generate the embedding
                embedding = genai.embed_content(
                    model=embedding_model,
                    content=test_text,
                    task_type="retrieval_document"
                )
                
                # Get the dimensions
                vector = embedding["embedding"]
                dimensions = len(vector)
                
                st.success(f"✅ Generated embedding with {dimensions} dimensions")
                st.write(f"First 5 values: {vector[:5]}")
                
                # Show dimensions info for database setup
                st.info(f"Your vector columns should be of type vector({dimensions})")
            else:
                st.error("Please configure Gemini API key first")
        except Exception as e:
            st.error(f"Embedding generation failed: {str(e)}")


# SQL Setup Helper
def show_sql_setup():
    st.title("SQL Setup Helper")
    
    st.markdown("""
    This page helps you set up the required SQL functions in your Supabase database for the RAG system to work.
    """)
    
    if "supabase_client" not in st.session_state:
        st.warning("Please connect to your Supabase database first.")
        return
        
    with st.expander("Step 1: Enable pgvector extension", expanded=True):
        st.markdown("First, you need to enable the pgvector extension in your Supabase database.")
        pgvector_code = """
        -- Enable the pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        """
        st.code(pgvector_code, language="sql")
        
        if st.button("Run pgvector setup"):
            try:
                # Use rpc instead of direct query
                response = st.session_state.supabase_client.rpc(
                    'exec_sql', 
                    {'sql_statement': 'CREATE EXTENSION IF NOT EXISTS vector;'}
                ).execute()
                st.success("✅ pgvector extension enabled successfully!")
            except Exception as e:
                st.error(f"Failed to enable pgvector: {str(e)}")
                st.info("Alternative approach: Use the Supabase SQL Editor in your dashboard to run this command manually.")
    
    with st.expander("Step 2: Create match_documents function", expanded=True):
        st.markdown("Next, you need to create the match_documents function for vector search.")
        match_documents_code = """
-- Function for matching documents based on embedding similarity
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector,
  match_threshold float,
  match_count int,
  table_name text
)
RETURNS TABLE (
  id integer,
  content text,
  similarity float
)
LANGUAGE plpgsql
AS $
DECLARE
  query_text TEXT;
  id_column TEXT;
  text_column TEXT;
  embedding_column TEXT;
BEGIN
  -- Set the appropriate columns based on the table name
  CASE table_name
    WHEN 'charities' THEN
      id_column := 'charity_id';
      text_column := 'description';
      embedding_column := 'embedding';
    WHEN 'charity_news' THEN
      id_column := 'news_id';
      text_column := 'content';
      embedding_column := 'embedding';
    WHEN 'charity_people' THEN
      id_column := 'person_id';
      text_column := 'name';
      embedding_column := 'embedding';
    WHEN 'charity_impact_areas' THEN
      id_column := 'impact_id';
      text_column := 'impact_description';
      embedding_column := 'embedding';
    WHEN 'charity_highlights' THEN
      id_column := 'highlight_id';
      text_column := 'highlight_text';
      embedding_column := 'embedding';
    WHEN 'charity_financials' THEN
      id_column := 'financial_id';
      text_column := 'other_financial_metrics';
      embedding_column := 'embedding';
    WHEN 'charity_causes' THEN
      id_column := 'charity_id';
      text_column := 'generated_text'; -- This needs to be handled differently
      embedding_column := 'embedding';
    WHEN 'charity_social_media' THEN
      id_column := 'social_id';
      text_column := 'platform';
      embedding_column := 'embedding';
    WHEN 'causes' THEN
      id_column := 'cause_id';
      text_column := 'description';
      embedding_column := 'embedding';
    ELSE
      RAISE EXCEPTION 'Unsupported table: %', table_name;
  END CASE;

  -- Handle special case for charity_causes
  IF table_name = 'charity_causes' THEN
    query_text := format('
      WITH joined_data AS (
        SELECT 
          cc.%I,
          CONCAT(''Charity "'', c.name, ''" supports cause "'', cs.name, ''"'') as content,
          1 - (cc.%I <=> $1) as similarity
        FROM 
          %I cc
        JOIN 
          charities c ON cc.charity_id = c.charity_id
        JOIN 
          causes cs ON cc.cause_id = cs.cause_id
        WHERE 
          cc.%I IS NOT NULL
      )
      SELECT %I, content, similarity
      FROM joined_data
      WHERE similarity > $2
      ORDER BY similarity DESC
      LIMIT $3
    ', id_column, embedding_column, table_name, embedding_column, id_column);
  ELSE
    -- Construct and execute dynamic SQL query for other tables
    query_text := format('
      SELECT %I as id, %I as content, 
             1 - (%I <=> $1) as similarity
      FROM %I
      WHERE %I IS NOT NULL AND 1 - (%I <=> $1) > $2
      ORDER BY similarity DESC
      LIMIT $3
    ', id_column, text_column, embedding_column, table_name, embedding_column, embedding_column);
  END IF;
  
  RETURN QUERY EXECUTE query_text USING query_embedding, match_threshold, match_count;
END;
$;
        """
        st.code(match_documents_code, language="sql")
        
        st.info("⚠️ Due to the complexity of this function, you'll need to create it directly in Supabase's SQL Editor.")
        
        if st.button("Copy SQL to clipboard"):
            # This is a client-side operation, so we'll use JavaScript
            st.markdown(
                f"""
                <textarea id="sql-code" style="position: absolute; left: -9999px;">{match_documents_code}</textarea>
                <script>
                    const copyToClipboard = () => {{
                        const text = document.getElementById('sql-code').value;
                        navigator.clipboard.writeText(text)
                            .then(() => alert('SQL code copied to clipboard!'))
                            .catch(err => alert('Failed to copy: ' + err));
                    }};
                    copyToClipboard();
                </script>
                """,
                unsafe_allow_html=True
            )
            st.success("SQL code copied! Now paste and run it in your Supabase SQL Editor")
            
        st.markdown("""
        ### Manual Instructions:
        1. Go to your [Supabase Dashboard](https://app.supabase.io)
        2. Select your project
        3. Go to the SQL Editor
        4. Paste the code above
        5. Click "Run" to execute
        """)
    
    with st.expander("Step 3: Create update_embedding_vector function", expanded=True):
        st.markdown("You need to create a function for updating embeddings.")
        update_embedding_code = """
CREATE OR REPLACE FUNCTION update_embedding_vector(
    p_table_name text,
    p_id_column text,
    p_id_value integer,
    p_embedding_column text,
    p_embedding float[]
) RETURNS void AS $
DECLARE
    query text;
BEGIN
    query := format('UPDATE %I SET %I = $1 WHERE %I = $2',
        p_table_name, p_embedding_column, p_id_column);
    EXECUTE query USING p_embedding, p_id_value;
END;
$ LANGUAGE plpgsql;
        """
        st.code(update_embedding_code, language="sql")
        
        if st.button("Copy SQL function to clipboard"):
            st.markdown(
                f"""
                <textarea id="update-sql-code" style="position: absolute; left: -9999px;">{update_embedding_code}</textarea>
                <script>
                    const copyUpdateSqlToClipboard = () => {{
                        const text = document.getElementById('update-sql-code').value;
                        navigator.clipboard.writeText(text)
                            .then(() => alert('SQL code copied to clipboard!'))
                            .catch(err => alert('Failed to copy: ' + err));
                    }};
                    copyUpdateSqlToClipboard();
                </script>
                """,
                unsafe_allow_html=True
            )
            st.success("SQL code copied! Now paste and run it in your Supabase SQL Editor")
                
    with st.expander("Step 4: Check vector columns", expanded=True):
        st.markdown("Make sure each table has a vector column for embeddings.")
        
        # List tables to check
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
        
        # Alternative approach using Supabase select
        for table in tables:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"Check **{table}** table for embedding column")
                
            with col2:
                if st.button(f"Check {table}", key=f"check_{table}"):
                    try:
                        # Use select query instead of raw SQL
                        result = st.session_state.supabase_client.table("information_schema.columns") \
                            .select("column_name", "data_type") \
                            .eq("table_name", table) \
                            .eq("column_name", "embedding") \
                            .execute()
                        
                        if result.data and len(result.data) > 0:
                            st.success(f"✅ {table} has an embedding column!")
                        else:
                            st.warning(f"⚠️ {table} needs an embedding column. Add it with this SQL:")
                            st.code(f"ALTER TABLE {table} ADD COLUMN embedding vector;", language="sql")
                    except Exception as e:
                        st.error(f"Failed to check {table}: {str(e)}")


# Sidebar navigation with updated pages
def setup_navigation():
    st.sidebar.title("Navigation")
    pages = {
        "Main Dashboard": show_main_dashboard,
        "RAG Chat": show_rag_chat_interface,
        "Data Viewer": show_data_viewer,
        "Embedding Generator": show_embedding_generator,
        "SQL Setup": show_sql_setup,
        "Settings": show_settings
    }

    # Add debug mode toggle in sidebar
    with st.sidebar.expander("Developer Options"):
        debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        st.session_state.debug_mode = debug_mode

    selected_page = st.sidebar.radio("Go to", list(pages.keys()))
    st.session_state.page = selected_page

    # Display the selected page
    pages[selected_page]()

# Main app entry point
def main():
    setup_navigation()
    
    # Add footer
    st.markdown("---")
    st.caption("Powered by Google Gemini 1.5 Flash with RAG")

if __name__ == "__main__":
    main()