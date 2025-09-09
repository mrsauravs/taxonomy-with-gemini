import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json
import google.generativeai as genai

# --- Page and Gemini API Configuration ---
st.set_page_config(layout="wide", page_title="AI Documentation Tagger")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    st.error("🚨 Gemini API Key not found! Please add it to your Streamlit secrets.")
    st.stop()

# --- Core Functions (unchanged) ---
def scrape_page_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.body
        return ' '.join(main_content.get_text(separator=' ', strip=True).split())
    except requests.exceptions.RequestException as e:
        return f"Scraping Error: {e}"

def scrape_deployment_tags(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        tags_container = soup.select_one('div.deployment-tags') or soup.select_one('div.topic-tags')
        if tags_container:
            tags = [tag.get_text(strip=True) for tag in tags_container.find_all('span')]
            return ", ".join(tags) if tags else "Tags Not Found"
        return "Tags Container Missing"
    except requests.exceptions.RequestException:
        return "Scraping Error"

def get_ai_analysis(content, user_roles, functional_areas, topics):
    if not content or "Scraping Error" in content:
        return {"error": content or "No content to analyze"}
    prompt = f"""
    Analyze the following documentation content...
    """ # NOTE: Prompt is collapsed for brevity but is unchanged
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_response)
    except Exception as e:
        return {"error": f"API or JSON Parsing Error: {e}"}

# --- Streamlit User Interface ---
st.title("📄 AI Documentation Tagger (Troubleshooting Mode)")
st.info("Upload your .txt files. Diagnostic messages will appear below as the process runs.")

col1, col2 = st.columns(2)
with col1:
    uploaded_urls = st.file_uploader("1. Upload URLs (.txt)", type="txt")
    uploaded_roles = st.file_uploader("2. Upload User Roles (.txt)", type="txt")
with col2:
    uploaded_areas = st.file_uploader("3. Upload Functional Areas (.txt)", type="txt")
    uploaded_topics = st.file_uploader("4. Upload Topics (.txt)", type="txt")

if st.button("🚀 Start Analysis", type="primary"):
    if all([uploaded_urls, uploaded_roles, uploaded_areas, uploaded_topics]):
        
        urls = [line for line in uploaded_urls.getvalue().decode("utf-8").strip().splitlines() if line.strip()]
        user_roles_list = uploaded_roles.getvalue().decode("utf-8")
        functional_areas_list = uploaded_areas.getvalue().decode("utf-8")
        topics_list = uploaded_topics.getvalue().decode("utf-8")
        
        # --- DIAGNOSTIC STEP 1: Check if files were read correctly ---
        st.info("--- DIAGNOSTIC LOG ---")
        st.write(f"✅ Found {len(urls)} URLs to process.")
        if not urls:
            st.error("Error: The URL list is empty. Please check your urls.txt file for content and formatting.")
            st.stop()
        
        st.write(f"✅ User Roles file loaded with {len(user_roles_list)} characters.")
        st.write(f"✅ Functional Areas file loaded with {len(functional_areas_list)} characters.")
        st.write(f"✅ Topics file loaded with {len(topics_list)} characters.")
        
        results = []
        progress_bar = st.progress(0, text="Starting analysis...")

        for i, url in enumerate(urls):
            st.markdown("---") # Separator for each URL
            progress_text = f"Processing ({i+1}/{len(urls)}): {url}"
            progress_bar.progress((i + 1) / len(urls), text=progress_text)
            
            # --- DIAGNOSTIC STEP 2: Check scraping results ---
            st.write(f"**URL {i+1}:** {url}")
            content = scrape_page_content(url)
            deployment_type = scrape_deployment_tags(url) 
            st.write(f"   - Deployment Tags Scraped: `{deployment_type}`")
            st.write(f"   - Main Content Scraped: `{len(content)} characters`")
            
            # --- DIAGNOSTIC STEP 3: Check AI response ---
            analysis = get_ai_analysis(content, user_roles_list, functional_areas_list, topics_list)
            st.write(f"   - AI Response Received: `{analysis}`")

            results.append({
                'Page URL': url, 'Deployment Type': deployment_type,
                'Keywords': analysis.get('keywords', analysis.get('error', 'N/A')),
                'User Role': analysis.get('user_roles', 'N/A'),
                'Functional Area': analysis.get('functional_areas', 'N/A'),
                'Topics': analysis.get('topics', 'N/A')
            })
            
            time.sleep(4) 
        
        st.markdown("---")
        progress_bar.empty()
        st.success("✅ Analysis Complete!")
        
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
