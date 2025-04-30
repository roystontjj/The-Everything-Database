import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import traceback

# Import utility functions and config
from config import get_config
from rag_utils import initialize_gemini_api, generate_embedding

def show_embedding_generator():
    """
    Shows the embedding generator interface adapted for CharityFinder
    """
    st.title("CharityFinder Embeddings Generator")
    
    # Check if API keys and client are set up
    if not st.session_state.get("gemini_api_key") or not st.session_state.get("supabase_client"):
        st.warning("Please configure your API keys in Settings before proceeding.")
        return
    
    # Initialize embedding model
    embedding_model = initialize_gemini_api()
    if not embedding_model:
        st.error("Failed to initialize embedding model. Please check your API key.")
        return
    
    # Define tables and their configurations - simplified for CharityFinder
    TABLE_CONFIGS = {
        "charities": {
            "id_column": "id",
            "text_column": "activities", # Using activities as the main text field for embeddings
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
        
        # Process the table
        process_table(table_name, config, batch_size, logs_placeholder)
            
        # Update statistics
        update_table_statistics(table_configs, stats_placeholder)
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    log_message("Embedding generation completed!", logs_placeholder)


def process_table(table_name, config, batch_size, logs_placeholder):
    """Process a table to generate embeddings"""
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