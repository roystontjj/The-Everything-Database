import streamlit as st
import google.generativeai as genai
import pandas as pd
from supabase import create_client

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
st.markdown("Ask questions, get creative content, or request assistance with various tasks.")

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
            # Get response from Gemini
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

# Charity Database Explorer
st.markdown("---")
st.title("Charity Database Explorer")

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
                financials = execute_query("charity_financials", filters={"charity_id": charity_id})
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

# NEW SECTION: Database Explorer with Gemini
st.markdown("---")
st.header("Ask About Charity Database")
db_question = st.text_input("What would you like to know about the charities?")

if st.button("Get Answer") and db_question:
    with st.spinner("Analyzing database..."):
        # Fetch relevant data from database
        all_charities = execute_query("charities")
        highlights = execute_query("charity_highlights", limit=20)
        impacts = execute_query("charity_impact_areas", limit=20)
        financials = execute_query("charity_financials")
        
        # Create data summary for the prompt
        data_context = f"""
        Database contains {len(all_charities) if all_charities is not None else 0} charities.
        
        SAMPLE CHARITY DATA:
        {all_charities.head(3).to_string() if all_charities is not None and not all_charities.empty else "No data"}
        
        SAMPLE HIGHLIGHTS:
        {highlights.head(3).to_string() if highlights is not None and not highlights.empty else "No data"}
        
        SAMPLE IMPACTS:
        {impacts.head(3).to_string() if impacts is not None and not impacts.empty else "No data"}
        
        SAMPLE FINANCIALS:
        {financials.head(3).to_string() if financials is not None and not financials.empty else "No data"}
        """
        
        # Create prompt for Gemini
        prompt = f"""
        You are analyzing a charity database with the following structure:
        - charities: Information about each charity (name, description, founded_year)
        - charity_highlights: Key highlights for each charity
        - charity_impact_areas: Areas where each charity creates impact
        - charity_financials: Financial metrics for charities
        
        Here is sample data from the database:
        {data_context}
        
        Based on this database, please answer: {db_question}
        """
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        st.subheader("Database Analysis")
        st.markdown(response.text)

# NEW SECTION: Advanced Database Analysis
st.markdown("---")
st.header("Advanced Database Analysis")
analysis_options = st.radio(
    "Choose analysis type:",
    ["Charity Comparison", "Financial Analysis", "Impact Assessment", "Custom Query"]
)

if analysis_options == "Charity Comparison":
    charities_list = charities['name'].tolist() if charities is not None and not charities.empty else []
    if len(charities_list) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            charity1 = st.selectbox("First charity:", charities_list, key="charity1")
        with col2:
            charity2 = st.selectbox("Second charity:", [c for c in charities_list if c != charity1], key="charity2")
        
        if st.button("Compare Charities"):
            with st.spinner("Analyzing charities..."):
                comparison = f"""
                Compare these two charities in detail, highlighting their strengths, weaknesses, 
                impact areas, and financial efficiency. Provide a recommendation on which one 
                might be more effective based on the data.
                
                {query_database_with_gemini(charity_name=charity1, topic="overview")}
                
                {query_database_with_gemini(charity_name=charity2, topic="overview")}
                """
                response = model.generate_content(comparison)
                st.markdown(response.text)
    else:
        st.info("Need at least two charities in the database for comparison.")
        
elif analysis_options == "Financial Analysis":
    if st.button("Analyze Financial Data"):
        with st.spinner("Analyzing financial data..."):
            financials = execute_query("charity_financials")
            all_charities = execute_query("charities")
            
            # Join the data
            if financials is not None and not financials.empty and all_charities is not None and not all_charities.empty:
                merged = pd.merge(
                    financials, 
                    all_charities[['charity_id', 'name']], 
                    on='charity_id', 
                    how='left'
                )
                
                prompt = f"""
                Analyze the financial data for these charities:
                
                {merged.to_string()}
                
                Provide insights on:
                1. Which charities appear most financially efficient
                2. Any concerning patterns in the data
                3. Recommendations for donors based on financial metrics
                4. How the financial metrics correlate with the charities' missions
                """
                
                response = model.generate_content(prompt)
                st.markdown(response.text)
            else:
                st.error("Could not retrieve financial data or charity data.")

elif analysis_options == "Impact Assessment":
    if st.button("Assess Overall Impact"):
        with st.spinner("Analyzing impact areas..."):
            # Get all impact areas and charity info
            impacts = execute_query("charity_impact_areas")
            all_charities = execute_query("charities")
            
            if impacts is not None and not impacts.empty and all_charities is not None and not all_charities.empty:
                # Join the data
                merged = pd.merge(
                    impacts,
                    all_charities[['charity_id', 'name']],
                    on='charity_id',
                    how='left'
                )
                
                prompt = f"""
                Analyze the impact areas for all charities in the database:
                
                {merged.to_string()}
                
                Provide insights on:
                1. What are the most common impact areas addressed by these charities?
                2. Are there any underserved areas that few charities are addressing?
                3. Which charities appear to have the most comprehensive approach to their mission?
                4. Recommendations for potential areas where new charities could focus
                """
                
                response = model.generate_content(prompt)
                st.markdown(response.text)
            else:
                st.error("Could not retrieve impact data or charity data.")

elif analysis_options == "Custom Query":
    custom_topic = st.text_input("Enter your custom analysis topic:")
    if st.button("Run Custom Analysis") and custom_topic:
        with st.spinner("Running custom analysis..."):
            response_text = query_database_with_gemini(topic=custom_topic)
            st.markdown(response_text)