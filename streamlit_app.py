import streamlit as st
import google.generativeai as genai
import pandas as pd
from supabase import create_client
import requests
import json
import time
import traceback
from typing import List, Dict, Any

# Import utility functions and config
from config import get_config
from rag_utils import initialize_gemini_api, generate_embedding, vector_search, format_results_for_rag
from embedding_generator import show_embedding_generator

# Page config and styling
st.set_page_config(page_title="CharityFinder RAG Assistant", layout="wide")

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

# Get configuration from config.py
config = get_config()

# Initialize session state from config
if config["gemini_api_key"] and not st.session_state.gemini_api_key:
    st.session_state.gemini_api_key = config["gemini_api_key"]
if config["supabase_url"] and not st.session_state.supabase_url:
    st.session_state.supabase_url = config["supabase_url"]
if config["supabase_key"] and not st.session_state.supabase_key:
    st.session_state.supabase_key = config["supabase_key"]
if config["debug_mode"] and not st.session_state.debug_mode:
    st.session_state.debug_mode = config["debug_mode"]

# Initialize Gemini API
try:
    if st.session_state.gemini_api_key:
        api_key = st.session_state.gemini_api_key
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini model
        if "chat" not in st.session_state and api_key:
            model = genai.GenerativeModel('gemini-1.5-flash')
            st.session_state.model = model
            st.session_state.chat = model.start_chat(history=[])
except Exception as e:
    # We'll handle this in the UI
    pass

# Initialize Supabase client
try:
    if st.session_state.supabase_url and st.session_state.supabase_key:
        supabase_url = st.session_state.supabase_url
        supabase_key = st.session_state.supabase_key
        
        supabase = create_client(supabase_url, supabase_key)
        st.session_state.supabase_client = supabase
except Exception as e:
    # We'll handle this in the UI
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
    st.title("CharityFinder Dashboard")
    st.markdown("Welcome to the CharityFinder Dashboard with RAG (Retrieval Augmented Generation).")
    
    # Display some basic statistics or information
    st.subheader("Database Overview")
    
    # If we have a Supabase connection, show basic stats
    if "supabase_client" in st.session_state:
        try:
            charities = execute_query("charities")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Charities", len(charities) if charities is not None and not charities.empty else 0)
            with col2:
                # Check if the sector column exists
                if charities is not None and not charities.empty and 'sector' in charities.columns:
                    sectors = charities['sector'].nunique()
                    st.metric("Sectors", sectors)
                else:
                    st.metric("Sectors", "N/A")
            with col3:
                # Count records with missing embeddings
                if charities is not None and not charities.empty and 'embedding' in charities.columns:
                    missing_embeddings = charities['embedding'].isna().sum()
                    st.metric("Missing Embeddings", missing_embeddings)
                else:
                    st.metric("Missing Embeddings", "N/A")
            
            # Display RAG status
            st.subheader("RAG System Status")
            
            # Check if the match_documents function exists
            try:
                # Try a simple vector search as a test
                test_query = "test query"
                embedding_model = initialize_gemini_api()
                query_embedding = generate_embedding(
                    test_query,
                    embedding_model,
                    task_type="retrieval_query"
                )
                
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
      id_column := 'id';
      text_column := 'activities'; -- We're using activities as our main text field
      embedding_column := 'embedding';
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
                
            # Display a sample of the charities data
            if charities is not None and not charities.empty:
                st.subheader("Sample Charities Data")
                st.dataframe(charities.head(5))
                
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
    
    # Select table to view (only charities in this version)
    tables = ["charities"]
    
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

# RAG CHAT INTERFACE
def show_rag_chat_interface():
    """
    RAG chat interface with query processing and debugging
    """
    st.title("💬 CharityFinder Assistant")
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
                    # Use the RAG query function
                    start_time = time.time()
                    debug_info = {}
                    
                    # Step 1: Retrieve relevant documents based on vector similarity
                    retrieval_time = time.time()
                    search_results = vector_search(
                        st.session_state.supabase_client,
                        user_prompt,
                        top_k=num_results,
                        threshold=similarity_threshold
                    )
                    
                    debug_info["retrieval_time"] = time.time() - retrieval_time
                    debug_info["num_results"] = len(search_results)
                    
                    # Step 2: Format retrieved documents into context
                    context_time = time.time()
                    formatted_context = format_results_for_rag(search_results, user_prompt)
                    debug_info["context_time"] = time.time() - context_time
                    
                    # Step 3: Create the RAG prompt with query and context
                    prompt_time = time.time()
                    rag_prompt = f"""
                    You are a helpful charity information assistant. Answer the user's question based on the retrieved context below.
                    If the question cannot be answered based on the context provided, say so, but try to be helpful using general 
                    knowledge about charities.
                    
                    CONTEXT:
                    {formatted_context}
                    
                    USER QUESTION: {user_prompt}
                    
                    Your answer should be comprehensive and directly address the user's question.
                    Include specific details from the context when available.
                    Format your answer nicely with appropriate paragraphs and sections when needed.
                    """
                    debug_info["prompt_time"] = time.time() - prompt_time
                    
                    # Step 4: Generate the final answer using Gemini
                    generation_time = time.time()
                    response = st.session_state.model.generate_content(rag_prompt)
                    debug_info["generation_time"] = time.time() - generation_time
                    
                    # Step 5: Prepare the final response
                    total_time = time.time() - start_time
                    
                    answer = response.text
                    context_html = formatted_context.replace("\n", "<br>")
                    
                    # Add assistant response to chat history with context
                    response_message = {
                        "role": "assistant", 
                        "content": answer,
                        "context": context_html,
                        "debug_info": debug_info,
                        "retrieval_time": f"{total_time:.2f} seconds"
                    }
                    
                    st.session_state.messages.append(response_message)
                    
                else:
                    # Regular chat without RAG
                    response = st.session_state.chat.send_message(user_prompt)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if st.session_state.debug_mode:
                    st.error(traceback.format_exc())
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

# Settings Page
def show_settings():
    st.title("Settings")
    
    st.subheader("Database Connection")
    
    # Supabase settings
    ##col1, col2 = st.columns(2)
    ##with col1:
        #supabase_url = st.text_input("Supabase URL", value=st.session_state.supabase_url, key="settings_supabase_url")
    ##with col2:
        #supabase_key = st.text_input("Supabase API Key", type="password", value=st.session_state.supabase_key, key="settings_supabase_key")
    
    # Gemini API settings    
    ##gemini_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key, key="settings_gemini_api_key")
    
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
                        "Show me information about charity work",
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
      id_column := 'id';
      text_column := 'activities'; -- Using activities as main text field
      embedding_column := 'embedding';
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
) RETURNS void AS $$
DECLARE
    query text;
BEGIN
    query := format('UPDATE %I SET %I = $1 WHERE %I = $2',
        p_table_name, p_embedding_column, p_id_column);
    EXECUTE query USING p_embedding, p_id_value;
END;
$$ LANGUAGE plpgsql;
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
        st.markdown("Make sure the charities table has a vector column for embeddings.")
        
        # List tables to check
        tables = ["charities"]
        
        # Check if embedding column exists
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

# Sidebar navigation
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
    st.caption("Powered by Google Gemini with RAG for CharityFinder")

if __name__ == "__main__":
    main()