import streamlit as st
import google.generativeai as genai
import psycopg2
import pandas as pd

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

# Function to connect to PostgreSQL (Supabase)
def connect_to_postgres():
    try:
        conn = psycopg2.connect(
            host=st.secrets["database"]["host"],
            database=st.secrets["database"]["name"],
            user=st.secrets["database"]["user"],
            password=st.secrets["database"]["password"],
            port=st.secrets["database"]["port"],
            sslmode='require'  # Required for Supabase
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# Function to test the connection
def test_connection():
    conn = connect_to_postgres()
    if conn:
        st.success("✅ Successfully connected to Supabase!")
        cursor = conn.cursor()
        cursor.execute("SELECT 'Connection is working!'")
        result = cursor.fetchone()
        st.write(result[0])
        conn.close()
        return True
    else:
        st.error("❌ Failed to connect to the database")
        return False

# Function to execute query
def execute_query(query, params=None):
    conn = connect_to_postgres()
    if conn:
        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Query execution error: {e}")
            conn.close()
            return None
    return None

# Charity Database Explorer
st.markdown("---")
st.title("Charity Database Explorer")

# Test connection
if st.button("Test Database Connection"):
    test_connection()

# Display charities
st.subheader("Charities in Database")
charities = execute_query("SELECT * FROM charities")

if charities is not None and not charities.empty:
    st.dataframe(charities)
    
    # Select a charity to view details
    charity_names = charities['name'].tolist()
    selected_charity = st.selectbox("Select a charity to view details:", charity_names)
    
    if selected_charity:
        charity_id = charities[charities['name'] == selected_charity]['charity_id'].iloc[0]
        
        # Create tabs for different sections
        tabs = st.tabs(["Overview", "Highlights", "Impact", "People", "Financials"])
        
        # Overview tab
        with tabs[0]:
            st.write(f"### {selected_charity}")
            charity_info = charities[charities['name'] == selected_charity].iloc[0]
            st.write(f"**Founded:** {charity_info['founded_year']}")
            st.write(f"**Website:** {charity_info['website']}")
            st.write(charity_info['description'])
        
        # Highlights tab
        with tabs[1]:
            highlights = execute_query(
                "SELECT highlight_text FROM charity_highlights WHERE charity_id = %s ORDER BY display_order",
                (charity_id,)
            )
            if highlights is not None and not highlights.empty:
                for i, row in highlights.iterrows():
                    st.write(f"✓ {row['highlight_text']}")
            else:
                st.write("No highlights found.")
        
        # Impact tab
        with tabs[2]:
            impact = execute_query(
                "SELECT impact_description FROM charity_impact_areas WHERE charity_id = %s ORDER BY display_order",
                (charity_id,)
            )
            if impact is not None and not impact.empty:
                for i, row in impact.iterrows():
                    st.write(f"{i+1}. {row['impact_description']}")
            else:
                st.write("No impact information found.")
        
        # Ask Gemini about this charity
        if st.button("Analyze with Gemini"):
            with st.spinner("Generating insights with Gemini..."):
                # Get all charity data
                query = """
                SELECT 
                    c.name, c.description, c.founded_year,
                    STRING_AGG(DISTINCT ci.impact_description, ' | ') as impact_areas,
                    STRING_AGG(DISTINCT ch.highlight_text, ' | ') as highlights,
                    cf.donation_income, cf.fundraising_efficiency, cf.program_expense_ratio
                FROM charities c
                LEFT JOIN charity_impact_areas ci ON c.charity_id = ci.charity_id
                LEFT JOIN charity_highlights ch ON c.charity_id = ch.charity_id
                LEFT JOIN charity_financials cf ON c.charity_id = cf.charity_id
                WHERE c.charity_id = %s
                GROUP BY c.charity_id, c.name, c.description, c.founded_year, 
                         cf.donation_income, cf.fundraising_efficiency, cf.program_expense_ratio
                """
                
                charity_data = execute_query(query, (charity_id,))
                
                if charity_data is not None and not charity_data.empty:
                    # Create prompt for Gemini
                    charity = charity_data.iloc[0]
                    prompt = f"""
                    Analyze this charity and provide insights:
                    
                    Name: {charity['name']}
                    Founded: {charity['founded_year']}
                    Description: {charity['description']}
                    
                    Impact Areas:
                    {charity['impact_areas']}
                    
                    Highlights:
                    {charity['highlights']}
                    
                    Financial Information:
                    - Donation Income: ${charity.get('donation_income', 'N/A')}
                    - Fundraising Efficiency: {charity.get('fundraising_efficiency', 'N/A')}%
                    - Program Expense Ratio: {charity.get('program_expense_ratio', 'N/A')}%
                    
                    Provide a comprehensive analysis of this charity's strengths, potential impact, and effectiveness.
                    """
                    
                    # Get analysis from Gemini
                    response = model.generate_content(prompt)
                    st.subheader("Gemini Analysis")
                    st.markdown(response.text)
                else:
                    st.error("Could not retrieve complete charity data for analysis")
else:
    st.info("No charities found in the database.")