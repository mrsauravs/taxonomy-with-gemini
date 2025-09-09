import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json
import google.generativeai as genai

# --- Page and Gemini API Configuration ---

# Set the layout and title for the Streamlit page
st.set_page_config(layout="wide", page_title="AI Documentation Tagger")

# Configure the Gemini API using Streamlit's secrets management
# This is the secure and recommended way for deployed apps.
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    st.error("🚨 Gemini API Key not found! Please add it to your Streamlit secrets.")
    st.stop()

# --- Core Functions ---

def scrape_page_content(url):
    """
    Scrapes the main textual content of a webpage for analysis.
    Returns the text content or an error string.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Prioritize specific content tags, falling back to the whole body
        main_content = soup.find('main') or soup.find('article') or soup.find(role='main') or soup.body
        
        # Extract text, replace multiple whitespaces with a single space
        return ' '.join(main_content.get_text(separator=' ', strip=True).split())
        
    except requests.exceptions.RequestException as e:
        return f"Scraping Error: {e}"

def get_ai_analysis(content, user_roles, functional_areas, topics):
    """
    Sends the scraped content to the Gemini API for a full analysis,
    requesting a structured JSON response.
    """
    if not content or "Scraping Error" in content:
        return {"error": content or "No content to analyze"}

    # This comprehensive prompt asks the AI to perform all tasks in one go.
    prompt = f"""
    Analyze the following documentation content. Your response MUST be a single, valid JSON object with four keys: "user_roles", "functional_areas", "topics", and "keywords".

    1.  **"user_roles"**: A string of the most relevant user roles, chosen ONLY from this list: {user_roles}
    2.  **"functional_areas"**: A string of the most relevant functional areas, chosen ONLY from this list: {functional_areas}
    3.  **"topics"**: A string of the most relevant topics, chosen ONLY from this list: {topics}
    4.  **"keywords"**: A string containing exactly 20 unique, comma-separated, high-value technical keywords from the content. Avoid generic terms like "documentation", "overview", or "introduction".

    **CONTENT TO ANALYZE** (first 8000 characters):
    "{content[:8000]}"

    **JSON RESPONSE**:
    """

    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON before parsing
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_response)
    except Exception as e:
        # Handle API errors or cases where the response is not valid JSON
        return {"error": f"API or JSON Parsing Error: {e}"}

# --- Streamlit User Interface ---

st.title("📄 AI-Powered Documentation Tagger & Keyword Generator")
st.info("Upload your .txt files. The app will scrape each URL and use the Gemini API to generate keywords and map categories.")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)
with col1:
    uploaded_urls = st.file_uploader("1. Upload URLs (.txt)", type="txt")
    uploaded_roles = st.file_uploader("2. Upload User Roles (.txt)", type="txt")
with col2:
    uploaded_areas = st.file_uploader("3. Upload Functional Areas (.txt)", type="txt")
    uploaded_topics = st.file_uploader("4. Upload Topics (.txt)", type="txt")

# Main action button to start the process
if st.button("🚀 Start Analysis", type="primary"):
    # Check if all necessary files have been uploaded
    if all([uploaded_urls, uploaded_roles, uploaded_areas, uploaded_topics]):
        
        # Read and decode the content from the uploaded text files
        urls = uploaded_urls.getvalue().decode("utf-8").strip().splitlines()
        user_roles_list = uploaded_roles.getvalue().decode("utf-8")
        functional_areas_list = uploaded_areas.getvalue().decode("utf-8")
        topics_list = uploaded_topics.getvalue().decode("utf-8")
        
        results = []
        progress_bar = st.progress(0, text="Starting analysis...")

        # Loop through each URL to scrape and analyze
        for i, url in enumerate(urls):
            progress_text = f"Processing ({i+1}/{len(urls)}): {url}"
            progress_bar.progress((i + 1) / len(urls), text=progress_text)
            
            content = scrape_page_content(url)
            analysis = get_ai_analysis(content, user_roles_list, functional_areas_list, topics_list)
            
            # Append the results for the current URL to our list
            results.append({
                'Page URL': url,
                'Keywords': analysis.get('keywords', analysis.get('error', 'N/A')),
                'User Role': analysis.get('user_roles', 'N/A'),
                'Functional Area': analysis.get('functional_areas', 'N/A'),
                'Topics': analysis.get('topics', 'N/A')
            })
            time.sleep(1) # Be respectful of API rate limits

        progress_bar.empty()
        st.success("✅ Analysis Complete!")
        
        # Display the results in a table and provide a download button
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="ai_analysis_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("Please upload all four required .txt files to begin.")