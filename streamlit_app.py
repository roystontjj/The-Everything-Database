import streamlit as st
import google.generativeai as genai
import pandas as pd
from supabase import create_client
import requests
import json
import time

# Import your modules
# You'll need to create rag_utils.py based on the previous code
try:
    from embedding_generator import show_embedding_generator
    from rag_utils import vector_search, format_results_for_rag, create_rag_prompt
except ImportError:
    st.error("Missing required modules. Make sure embedding_generator.py and rag_utils.py are in the same directory.")

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
    api_key = st.secrets["GEMINI_API_KEY"]
    st.session_state.gemini_api_key = api_key
    genai.configure(api_key=api_key)
    
    # Initialize the Gemini model
    if "chat" not in st.session_state:
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.model = model
        st.session_state.chat = model.start_chat(history=[])
except Exception as e:
    st.sidebar.error(f"Failed to initialize Gemini API: {e}")

# Initialize Supabase client
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    st.session_state.supabase_url = supabase_url
    st.session_state.supabase_key = supabase_key
    
    supabase = create_client(supabase_url, supabase_key)
    st.session_state.supabase_client = supabase
except Exception as e:
    st.sidebar.error(f"Failed to initialize Supabase client: {e}")

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

# Main Dashboard Page (Keep your original implementation)
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

# Data Viewer Page (Keep your original implementation)
def show_data_viewer():
    # Keep the current implementation
    st.title("Data Viewer")
    
    if "supabase_client" not in st.session_state:
        st.warning("Please connect to your Supabase database first.")
        return
    
    # The rest of your data viewer code...

# NEW RAG Chat Interface
def show_rag_chat_interface():
    st.title("💬 Charity RAG Assistant")
    st.markdown("Ask questions about charities using Retrieval Augmented Generation!")

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
        with st.spinner("Processing..."):
            try:
                if use_rag and "supabase_client" in st.session_state:
                    start_time = time.time()
                    
                    # Get relevant documents using vector search
                    search_results = vector_search(
                        st.session_state.supabase_client, 
                        user_prompt, 
                        top_k=num_results,
                        threshold=similarity_threshold
                    )
                    
                    retrieval_time = time.time() - start_time
                    
                    # Format the retrieved documents for context
                    formatted_context = format_results_for_rag(search_results, user_prompt)
                    
                    # Create prompt with RAG context
                    rag_prompt = create_rag_prompt(user_prompt, formatted_context)
                    
                    # Get response from Gemini with RAG context
                    response = st.session_state.model.generate_content(rag_prompt)
                    
                    # Add assistant response to chat history with context
                    context_html = formatted_context.replace("\n", "<br>")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.text,
                        "context": context_html,
                        "retrieval_time": f"{retrieval_time:.2f} seconds"
                    })
                    
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

# Utility to check SQL functions
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
                response = st.session_state.supabase_client.query(pgvector_code).execute()
                st.success("✅ pgvector extension enabled successfully!")
            except Exception as e:
                st.error(f"Failed to enable pgvector: {str(e)}")
    
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
        
        if st.button("Create match_documents function"):
            try:
                response = st.session_state.supabase_client.query(match_documents_code).execute()
                st.success("✅ match_documents function created successfully!")
            except Exception as e:
                st.error(f"Failed to create match_documents function: {str(e)}")
    
    with st.expander("Step 3: Check update_embedding_vector function", expanded=True):
        st.markdown("You already have an update_embedding_vector function for updating embeddings, but let's check it.")
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
        
        if st.button("Check/Create update_embedding_vector function"):
            try:
                # First check if function exists
                check_query = """
                SELECT 1 FROM pg_proc WHERE proname = 'update_embedding_vector';
                """
                check_result = st.session_state.supabase_client.query(check_query).execute()
                
                if check_result.data and len(check_result.data) > 0:
                    st.success("✅ update_embedding_vector function already exists!")
                else:
                    # Create the function
                    response = st.session_state.supabase_client.query(update_embedding_code).execute()
                    st.success("✅ update_embedding_vector function created successfully!")
            except Exception as e:
                st.error(f"Failed to check/create update_embedding_vector function: {str(e)}")
                
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
        
        for table in tables:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"Check **{table}** table for embedding column")
                
            with col2:
                if st.button(f"Check {table}", key=f"check_{table}"):
                    try:
                        # Query to check if embedding column exists
                        check_query = f"""
                        SELECT column_name, data_type
                        FROM information_schema.columns 
                        WHERE table_name = '{table}' AND column_name = 'embedding';
                        """
                        check_result = st.session_state.supabase_client.query(check_query).execute()
                        
                        if check_result.data and len(check_result.data) > 0:
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