import streamlit as st
import google.generativeai as genai
import pandas as pd
from supabase import create_client
import requests
import json

# Page config and styling
st.set_page_config(page_title="Gemini AI Assistant", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# App title
st.title("💬 Gemini AI Assistant")
st.markdown("Ask questions, get creative content, or request assistance with various tasks. You can also ask about the charity database!")

# Get API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Initialize Supabase client
try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
    st.session_state.supabase_client = supabase
except Exception as e:
    st.error(f"Failed to initialize Supabase client: {e}")

# Function to test the Supabase connection
def test_connection():
    try:
        # Just fetch a small amount of data to test the connection
        response = supabase.table('charities').select('*').limit(1).execute()
        st.success("✅ Successfully connected to Supabase!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to connect to Supabase API: {e}")
        return False

# Function to execute query via Supabase API
def execute_query(table_name, select_columns="*", filters=None, order_by=None, limit=None):
    try:
        # Start with a basic query
        query = supabase.table(table_name).select(select_columns)
        
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

# Function to query database with Gemini
def query_database_with_gemini(charity_name=None, topic=None):
    """Query the database with Gemini based on specific parameters"""
    data_to_include = {}
    
    if charity_name:
        # Get specific charity data
        charity = execute_query("charities", filters={"name": charity_name})
        if charity is not None and not charity.empty:
            charity_id = charity.iloc[0]['charity_id']
            data_to_include["charity"] = charity
            data_to_include["highlights"] = execute_query("charity_highlights", 
                                                        filters={"charity_id": charity_id})
            data_to_include["impacts"] = execute_query("charity_impact_areas", 
                                                    filters={"charity_id": charity_id})
            data_to_include["financials"] = execute_query("charity_financials", 
                                                        filters={"charity_id": charity_id})
    else:
        # Get general database data
        data_to_include["charities"] = execute_query("charities")
        data_to_include["highlights"] = execute_query("charity_highlights", limit=10)
        data_to_include["impacts"] = execute_query("charity_impact_areas", limit=10)
        data_to_include["financials"] = execute_query("charity_financials")
    
    # Format the data
    formatted_data = ""
    for key, data in data_to_include.items():
        if data is not None and not data.empty:
            formatted_data += f"\n{key.upper()} DATA:\n{data.to_string()}\n"
    
    # Create prompt based on topic or general inquiry
    if topic:
        prompt = f"Analyze this charity database data focusing on {topic}:\n{formatted_data}"
    else:
        prompt = f"Analyze this charity database data and provide insights:\n{formatted_data}"
    
    # Get response from Gemini
    return model.generate_content(prompt).text

# Function to create database context for prompts
def get_database_context():
    # Fetch data from database
    all_charities = execute_query("charities")
    highlights = execute_query("charity_highlights", limit=5)
    impacts = execute_query("charity_impact_areas", limit=5)
    financials = execute_query("charity_financials")
    
    # Create context string
    context = f"""
    DATABASE CONTEXT:
    Database contains {len(all_charities) if all_charities is not None and not all_charities.empty else 0} charities.
    
    SAMPLE CHARITY DATA:
    {all_charities.head(3).to_string() if all_charities is not None and not all_charities.empty else "No data"}
    
    SAMPLE HIGHLIGHTS:
    {highlights.head(3).to_string() if highlights is not None and not highlights.empty else "No data"}
    
    SAMPLE IMPACTS:
    {impacts.head(3).to_string() if impacts is not None and not impacts.empty else "No data"}
    
    SAMPLE FINANCIALS:
    {financials.head(3).to_string() if financials is not None and not financials.empty else "No data"}
    """
    
    return context

# Display chat history
st.subheader("Conversation")
chat_container = st.container(height=400)
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><strong>You:</strong><br>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong><br>{message['content']}</div>", unsafe_allow_html=True)

# User input section
st.subheader("Your Question")
with st.form(key="prompt_form", clear_on_submit=True):
    user_prompt = st.text_area("Type your message here:", height=100, key="prompt_input")
    
    # Add a checkbox to explicitly query the database
    include_db_context = st.checkbox("Include charity database context", value=False, 
                                    help="Check this if your question is about the charity database")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.form_submit_button("Send", use_container_width=True)
    with col2:
        if st.form_submit_button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat = model.start_chat(history=[])
            st.rerun()

# Process the user input when the form is submitted
if submit_button and user_prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Show a spinner while processing
    with st.spinner("Thinking..."):
        try:
            # Check if this is a database-related question or user selected the checkbox
            database_keywords = ["database", "charity", "charities", "data", "compare", 
                               "impact", "financial", "highlights", "donation"]
            
            is_database_question = include_db_context or any(keyword in user_prompt.lower() for keyword in database_keywords)
            
            if is_database_question:
                # Get database context
                data_context = get_database_context()
                
                # Create prompt with database context
                db_prompt = f"""
                You are connected to a charity database with information about various charities.
                Here's some sample data from the database:
                {data_context}
                
                User question: {user_prompt}
                
                When answering, make use of the database information provided above.
                """
                
                # Get response from Gemini with database context
                response = model.generate_content(db_prompt)
            else:
                # Regular chat without database context
                response = st.session_state.chat.send_message(user_prompt)
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
    
    # Rerun to update the UI with the new messages
    st.rerun()

# Add some footer information
st.markdown("---")
st.caption("Powered by Google Gemini 1.5 Flash")

# Database connection test and Charity Database Explorer
with st.expander("Charity Database Explorer", expanded=False):
    # Test connection
    if st.button("Test Database Connection"):
        test_connection()

    # Display charities
    st.subheader("Charities in Database")
    charities = execute_query("charities")

    if charities is not None and not charities.empty:
        st.dataframe(charities)
        
        # Select a charity to view details
        charity_names = charities['name'].tolist()
        selected_charity = st.selectbox("Select a charity to view details:", charity_names)
        
        if selected_charity:
            charity_id = charities[charities['name'] == selected_charity]['charity_id'].iloc[0]
            
            # Create tabs for different sections
            tabs = st.tabs(["Overview", "Highlights", "Impact", "Financials"])
            
            # Overview tab
            with tabs[0]:
                st.write(f"### {selected_charity}")
                charity_info = charities[charities['name'] == selected_charity].iloc[0]
                st.write(f"**Founded:** {charity_info['founded_year']}")
                st.write(f"**Website:** {charity_info['website']}")
                st.write(charity_info['description'])
            
            # Highlights tab
            with tabs[1]:
                highlights = execute_query("charity_highlights", filters={"charity_id": charity_id}, order_by="display_order")
                if highlights is not None and not highlights.empty:
                    for i, row in highlights.iterrows():
                        st.write(f"✓ {row['highlight_text']}")
                else:
                    st.write("No highlights found.")
            
            # Impact tab
            with tabs[2]:
                impact = execute_query("charity_impact_areas", filters={"charity_id": charity_id}, order_by="display_order")
                if impact is not None and not impact.empty:
                    for i, row in impact.iterrows():
                        st.write(f"{i+1}. {row['impact_description']}")
                else:
                    st.write("No impact information found.")
            
            # Financials tab
            with tabs[3]:
                financials = execute_query("charity_financials", filters={"charity_id": charity_id})
                if financials is not None and not financials.empty:
                    st.write(f"**Donation Income:** ${financials.iloc[0]['donation_income']:,.2f}")
                    st.write(f"**Fundraising Efficiency:** {financials.iloc[0]['fundraising_efficiency']}%")
                    st.write(f"**Program Expense Ratio:** {financials.iloc[0]['program_expense_ratio']}%")
                else:
                    st.write("No financial information found.")
            
            # Ask Gemini about this charity
            if st.button("Analyze with Gemini"):
                with st.spinner("Generating insights with Gemini..."):
                    # Get all charity data
                    charity_data = {}
                    charity_data.update(charity_info.to_dict())
                    
                    # Add impact and highlights
                    if impact is not None and not impact.empty:
                        impact_text = " | ".join(impact['impact_description'].tolist())
                        charity_data['impact_areas'] = impact_text
                    
                    if highlights is not None and not highlights.empty:
                        highlights_text = " | ".join(highlights['highlight_text'].tolist())
                        charity_data['highlights'] = highlights_text
                    
                    # Get financials if available
                    if financials is not None and not financials.empty:
                        charity_data.update(financials.iloc[0].to_dict())
                    
                    # Create prompt for Gemini
                    prompt = f"""
                    Analyze this charity and provide insights:
                    
                    Name: {charity_data.get('name', 'N/A')}
                    Founded: {charity_data.get('founded_year', 'N/A')}
                    Description: {charity_data.get('description', 'N/A')}
                    
                    Impact Areas:
                    {charity_data.get('impact_areas', 'N/A')}
                    
                    Highlights:
                    {charity_data.get('highlights', 'N/A')}
                    
                    Financial Information:
                    - Donation Income: ${charity_data.get('donation_income', 'N/A')}
                    - Fundraising Efficiency: {charity_data.get('fundraising_efficiency', 'N/A')}%
                    - Program Expense Ratio: {charity_data.get('program_expense_ratio', 'N/A')}%
                    
                    Provide a comprehensive analysis of this charity's strengths, potential impact, and effectiveness.
                    """
                    
                    # Get analysis from Gemini
                    response = model.generate_content(prompt)
                    st.subheader("Gemini Analysis")
                    st.markdown(response.text)
    else:
        st.info("No charities found in the database.")

# Add this function to your Streamlit app to make gemini test call
def test_gemini_embedding_dimension(api_key):
    try:
        st.write("Testing Gemini embedding dimension...")
        
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-embedding-exp-03-07:embedContent?key={api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "This is a test sentence to determine embedding dimension."
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        embedding = data["embeddings"][0]["values"]
        
        dimension_size = len(embedding)
        st.success(f"Embedding dimension size: {dimension_size}")
        st.write(f"Sample of first few values: {embedding[:5]}")
        
        return dimension_size
    except Exception as e:
        st.error(f"Error testing Gemini embedding dimension: {e}")
        return None

# You can call this function in your app
# For example:
if st.button("Test Gemini Embedding Dimension"):
    api_key = st.secrets["GEMINI_API_KEY"]  # Store your API key in Streamlit secrets
    dimension_size = test_gemini_embedding_dimension(api_key)
    if dimension_size:
        st.session_state.dimension_size = dimension_size  # Save it for later use