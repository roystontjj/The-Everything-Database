import os
import streamlit as st

def get_config():
    """
    Retrieve configuration from environment variables or Streamlit secrets.
    Returns a dictionary with configuration values.
    """
    config = {
        "supabase_url": None,
        "supabase_key": None,
        "gemini_api_key": None,
        "debug_mode": False,
    }
    
    # Try to get from Streamlit secrets
    try:
        if st.secrets.get("SUPABASE_URL"):
            config["supabase_url"] = st.secrets["SUPABASE_URL"]
        if st.secrets.get("SUPABASE_KEY"):
            config["supabase_key"] = st.secrets["SUPABASE_KEY"]
        if st.secrets.get("GEMINI_API_KEY"):
            config["gemini_api_key"] = st.secrets["GEMINI_API_KEY"]
        if st.secrets.get("DEBUG_MODE"):
            config["debug_mode"] = st.secrets["DEBUG_MODE"]
    except Exception:
        # If secrets aren't available, check environment variables
        pass
        
    # Try to get from environment variables (as a fallback)
    if not config["supabase_url"]:
        config["supabase_url"] = os.environ.get("SUPABASE_URL")
    if not config["supabase_key"]:
        config["supabase_key"] = os.environ.get("SUPABASE_KEY")
    if not config["gemini_api_key"]:
        config["gemini_api_key"] = os.environ.get("GEMINI_API_KEY")
    if not config["debug_mode"]:
        config["debug_mode"] = os.environ.get("DEBUG_MODE", "false").lower() == "true"
        
    return config