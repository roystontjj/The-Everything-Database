import streamlit as st
import pandas as pd
import json
import time
from supabase import create_client
import google.generativeai as genai
from typing import List, Dict, Any, Tuple

# Database schema
TABLE_CONFIGS = [
    {
        "name": "charities",
        "text_column": "description",
        "id_column": "charity_id",
        "embedding_column": "embedding"
    },
    {
        "name": "charity_news",
        "text_column": "content",
        "id_column": "news_id",
        "embedding_column": "embedding"
    },
    {
        "name": "charity_people",
        "text_column": "name",
        "id_column": "person_id",
        "embedding_column": "embedding"
    },
    {
        "name": "charity_impact_areas",
        "text_column": "impact_description",
        "id_column": "impact_id",
        "embedding_column": "embedding"
    },
    {
        "name": "charity_highlights",
        "text_column": "highlight_text",
        "id_column": "highlight_id",
        "embedding_column": "embedding"
    },
    {
        "name": "charity_financials",
        "text_column": "other_financial_metrics",
        "id_column": "financial_id",
        "embedding_column": "embedding"
    },
    {
        "name": "charity_causes",
        "text_column": None,  # This needs special handling as it doesn't have a clear text column
        "id_column": "charity_id",
        "embedding_column": "embedding"
    },
    {
        "name": "charity_social_media",
        "text_column": "platform",
        "id_column": "social_id",
        "embedding_column": "embedding"
    },
    {
        "name": "causes",
        "text_column": "description",
        "id_column": "cause_id",
        "embedding_column": "embedding"
    }
]

def connect_to_supabase():
    """Establish connection to Supabase"""
    try:
        client = create_client(st.session_state.supabase_url, st.session_state.supabase_key)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return None

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
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

def get_records_without_embeddings(supabase, table_config, limit=50) -> pd.DataFrame:
    """Fetch records that have empty embeddings"""
    try:
        result = supabase.table(table_config["name"]) \
            .select(f"{table_config['id_column']}, {table_config['text_column'] if table_config['text_column'] else '*'}") \
            .is_(table_config["embedding_column"], "null") \
            .limit(limit) \
            .execute()
        
        # Handle the case for charity_causes which requires joining with other tables
        if table_config["name"] == "charity_causes" and len(result.data) > 0:
            # For charity_causes, we'll need to join with charities and causes tables
            # to create meaningful text for embeddings
            enriched_data = []
            for row in result.data:
                charity_id = row["charity_id"]
                cause_id = row["cause_id"]
                
                # Get charity name
                charity_result = supabase.table("charities") \
                    .select("name") \
                    .eq("charity_id", charity_id) \
                    .execute()
                
                # Get cause name
                cause_result = supabase.table("causes") \
                    .select("name") \
                    .eq("cause_id", cause_id) \
                    .execute()
                
                if charity_result.data and cause_result.data:
                    charity_name = charity_result.data[0]["name"]
                    cause_name = cause_result.data[0]["name"]
                    row["generated_text"] = f"Charity '{charity_name}' supports the cause '{cause_name}'"
                    enriched_data.append(row)
            
            return pd.DataFrame(enriched_data)
        
        return pd.DataFrame(result.data)
    except Exception as e:
        st.error(f"Error fetching records from {table_config['name']}: {str(e)}")
        return pd.DataFrame()

def generate_embedding(text: str, model: str):
    """Generate embedding using Gemini API"""
    try:
        if not text or text.strip() == "":
            return None
        
        embedding = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        
        # Return the embedding values as a list
        return embedding["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        time.sleep(1)  # Add a small delay in case of rate limiting
        return None

def update_record_with_embedding(supabase, table_name: str, id_column: str, id_value: int, 
                                embedding_column: str, embedding_vector: List[float]):
    """Update a record with its new embedding"""
    try:
        supabase.table(table_name) \
            .update({embedding_column: embedding_vector}) \
            .eq(id_column, id_value) \
            .execute()
        return True
    except Exception as e:
        st.error(f"Error updating record in {table_name}: {str(e)}")
        return False

def process_table(supabase, table_config: Dict, embedding_model: str, batch_size: int):
    """Process a single table to update missing embeddings"""
    table_name = table_config["name"]
    id_column = table_config["id_column"]
    text_column = table_config["text_column"]
    embedding_column = table_config["embedding_column"]
    
    log_msg = f"Processing table: {table_name}"
    st.session_state.logs.append(log_msg)
    
    # Get records without embeddings
    df = get_records_without_embeddings(supabase, table_config, batch_size)
    
    if df.empty:
        log_msg = f"No records found with missing embeddings in {table_name}"
        st.session_state.logs.append(log_msg)
        return 0
    
    log_msg = f"Found {len(df)} records with missing embeddings in {table_name}"
    st.session_state.logs.append(log_msg)
    
    updated_count = 0
    for _, row in df.iterrows():
        # Get the appropriate text to embed
        if table_name == "charity_causes":
            text_to_embed = row.get("generated_text", "")
        else:
            text_to_embed = row.get(text_column, "")
        
        # Generate embedding
        if text_to_embed:
            embedding_vector = generate_embedding(text_to_embed, embedding_model)
            
            if embedding_vector:
                # Update the record
                id_value = row[id_column]
                success = update_record_with_embedding(
                    supabase, table_name, id_column, id_value, embedding_column, embedding_vector
                )
                
                if success:
                    updated_count += 1
                    log_msg = f"Updated embedding for {table_name} record with {id_column}={id_value}"
                    st.session_state.logs.append(log_msg)
                
                # Small delay to avoid hitting rate limits
                time.sleep(0.5)
    
    log_msg = f"Updated {updated_count} records in {table_name}"
    st.session_state.logs.append(log_msg)
    return updated_count

def run_embedding_update():
    """Main function to update embeddings for all tables"""
    if not st.session_state.supabase_url or not st.session_state.supabase_key:
        st.error("Please provide Supabase URL and API key")
        return
    
    # Check for Gemini API key
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        if not st.session_state.gemini_api_key:
            st.error("Please provide Gemini API key")
            return
    
    st.session_state.processing = True
    st.session_state.progress = 0
    st.session_state.logs = []
    
    # Connect to Supabase
    supabase = connect_to_supabase()
    if not supabase:
        st.session_state.processing = False
        return
    
    # Initialize Gemini API
    embedding_model = initialize_gemini_api()
    if not embedding_model:
        st.session_state.processing = False
        return
    
    total_tables = len(TABLE_CONFIGS)
    total_updated = 0
    
    # Process each table
    for i, table_config in enumerate(TABLE_CONFIGS):
        updated_count = process_table(supabase, table_config, embedding_model, st.session_state.batch_size)
        total_updated += updated_count
        
        # Update progress
        st.session_state.progress = (i + 1) / total_tables
    
    st.session_state.logs.append(f"Completed embedding updates. Updated {total_updated} records in total.")
    st.session_state.processing = False

def show_embedding_generator():
    """Main function to display the embedding generator page"""
    st.title("Database Embeddings Generator")
    
    # Initialize session state variables if they don't exist
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "supabase_url" not in st.session_state:
        st.session_state.supabase_url = ""
    if "supabase_key" not in st.session_state:
        st.session_state.supabase_key = ""
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""
    if "batch_size" not in st.session_state:
        st.session_state.batch_size = 10
    
    # Supabase credentials input
    with st.sidebar:
        st.header("Database Configuration")
        st.session_state.supabase_url = st.text_input("Supabase URL", value=st.session_state.supabase_url)
        st.session_state.supabase_key = st.text_input("Supabase API Key", type="password", value=st.session_state.supabase_key)
        
        # Streamlit secrets management for Gemini API
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            gemini_key_available = True
            st.success("Gemini API key found in Streamlit secrets.")
        except:
            gemini_key_available = False
            st.warning("Gemini API key not found in Streamlit secrets.")
            st.session_state.gemini_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key)
        
        st.session_state.batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=st.session_state.batch_size)
        
        st.markdown("---")
        st.markdown("""
        ### How it works
        This tool scans your Supabase database for records with empty embedding fields 
        and generates embeddings using the Gemini API.
        """)
    
    # Main app layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Button to start processing
        if not st.session_state.processing:
            if st.button("Generate Missing Embeddings", use_container_width=True):
                run_embedding_update()
        else:
            st.button("Processing...", disabled=True, use_container_width=True)
            st.progress(st.session_state.progress)
    
        # Display logs
        st.subheader("Processing Logs")
        log_container = st.container(height=400)
        with log_container:
            for log in st.session_state.logs:
                st.text(log)
    
    with col2:
        st.subheader("Table Statistics")
        
        # Only show statistics if we have Supabase credentials
        if st.session_state.supabase_url and st.session_state.supabase_key:
            supabase = connect_to_supabase()
            if supabase:
                stats_container = st.container(height=400)
                with stats_container:
                    for table_config in TABLE_CONFIGS:
                        try:
                            # Get total count
                            total_result = supabase.table(table_config["name"]) \
                                .select(table_config["id_column"], count="exact") \
                                .execute()
                            
                            # Get count of records without embeddings
                            missing_result = supabase.table(table_config["name"]) \
                                .select(table_config["id_column"], count="exact") \
                                .is_(table_config["embedding_column"], "null") \
                                .execute()
                            
                            total_count = total_result.count
                            missing_count = missing_result.count
                            
                            if total_count > 0:
                                completion_rate = ((total_count - missing_count) / total_count) * 100
                            else:
                                completion_rate = 100
                            
                            st.markdown(f"**{table_config['name']}**: {missing_count} missing out of {total_count} ({completion_rate:.1f}% complete)")
                        except Exception as e:
                            st.markdown(f"**{table_config['name']}**: Error fetching statistics")