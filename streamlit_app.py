import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import google.generativeai as genai
import re
import json
import time

# --- Configuration ---
STABLE_MODEL = "gemini-1.5-flash-latest"

# --- Utility and Scraping Functions ---

@st.cache_data
def analyze_page_content(url):
    """Fetches and parses a URL for title and main content analysis."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').get_text().strip() if soup.find('title') else 'No Title Found'
        return soup, title
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch {url}: {e}")
        return None, "Fetch Error"

def get_deployment_type_from_scraping(soup):
    """Determines deployment type from parsed HTML based on specific class labels."""
    if not soup: return ""
    has_cloud = soup.find('p', class_='cloud-label') is not None
    has_on_prem = soup.find('p', class_='on-prem-label') is not None
    if has_cloud and has_on_prem: return "Alation Cloud Service, Customer Managed"
    if has_cloud: return "Alation Cloud Service"
    if has_on_prem: return "Customer Managed"
    return ""

# --- LLM Analysis Functions using Gemini API ---

def call_llm_with_retry(model, prompt):
    """Calls the Gemini API with a stable model and retries on failure."""
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_message = str(e)
            if attempt < 2:
                st.warning(f"API call with {STABLE_MODEL} failed (attempt {attempt + 1}/3): {error_message}. Retrying...")
                time.sleep(5)
            else:
                st.error(f"All retries failed for model {STABLE_MODEL}: {error_message}")
    
    raise Exception("LLM API Error: All retries failed.")


def get_deployment_type_with_llm(model, soup):
    """Uses an LLM to infer deployment type if scraping fails."""
    if not soup: return "Analysis Error"
    main_content = soup.find('article') or soup.find('main') or soup.body
    content_text = main_content.get_text(separator=' ', strip=True)[:15000] if main_content else ""

    prompt = f"""You are an expert text classifier. Read the following content and determine if it applies to "Alation Cloud Service", "Customer Managed", or both. Your answer MUST be one of those three options ONLY, with no other text.
    Content: --- {content_text} ---
    Based on the content, the correct deployment type is:"""
    
    try:
        response = call_llm_with_retry(model, prompt)
        cleaned_response = response.strip()
        valid_responses = ["Alation Cloud Service", "Customer Managed", "Alation Cloud Service, Customer Managed"]
        return f"{cleaned_response} (Inferred by LLM)" if cleaned_response in valid_responses else "LLM Inference Failed"
    except Exception as e:
        return f"LLM API Error: {str(e)}"

def get_metadata_analysis_with_llm(model, soup, roles, areas, topics):
    """Uses an LLM for metadata mapping."""
    if not soup: return {}
    main_content = soup.find('article') or soup.find('main') or soup.body
    content_text = main_content.get_text(separator=' ', strip=True)[:15000] if main_content else ""

    prompt = f"""
    You are an expert content analyst. Analyze the provided technical documentation content.
    From the provided lists, select the MOST RELEVANT User Role(s), Functional Area(s), and Topic(s). Choose only the best fit for each category.

    **Available User Roles:** {', '.join(roles)}
    **Available Functional Areas:** {', '.join(areas)}
    **Available Topics:** {', '.join(topics)}

    **Content to Analyze:** --- {content_text} ---
    
    Provide your response in a single, clean JSON object format like this example, with no other text or formatting: {{"user_role": "Steward, Catalog Admin", "functional_area": "Data Quality", "topics": "Data Quality Monitors, Troubleshooting"}}
    """
    
    try:
        response_text = call_llm_with_retry(model, prompt)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"error": "Failed to parse LLM response"}
    except Exception as e:
        return {"error": f"LLM API Error: {str(e)}"}

def get_keywords_with_llm(model, soup, page_title):
    """Uses an LLM for keyword generation, with special logic for OCF Connectors."""
    if not soup: return {}
    main_content = soup.find('article') or soup.find('main') or soup.body
    content_text = main_content.get_text(separator=' ', strip=True)[:15000] if main_content else ""

    connector_instructions = ""
    if "OCF Connector" in page_title:
        connector_name = page_title.split('|')[0].strip()
        db_system = connector_name.replace("OCF Connector", "").strip()
        connector_instructions = f'**Critical Rule:** This page is about an OCF Connector. The keywords MUST include both "{connector_name}" and "{db_system} data source".'

    prompt = f"""
    You are an expert content analyst. Analyze the provided technical documentation content to generate exactly 20 unique, comma-separated technical keywords.
    **Exclusion Rules:** Exclude generic words like 'guide', 'documentation', 'button', 'click', 'data', 'alation', 'prerequisites', 'overview', 'steps'.
    {connector_instructions}

    **Content to Analyze:** --- {content_text} ---
    
    Provide your response in a single, clean JSON object format like this example, with no other text or formatting: {{"keywords": "keyword1, keyword2, ..., keyword20"}}
    """
    
    try:
        response_text = call_llm_with_retry(model, prompt)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"error": "Failed to parse LLM response"}
    except Exception as e:
        return {"error": f"LLM API Error: {str(e)}"}

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("📄 Intelligent Content Analysis Workflow")

st.sidebar.header("Configuration")
GOOGLE_API_KEY = st.sidebar.text_input("Enter your Google Gemini API Key", type="password")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(STABLE_MODEL)
        st.sidebar.success("API Key configured successfully!")
    except Exception as e:
        st.sidebar.error(f"Invalid API Key: {e}")
        GOOGLE_API_KEY = None
else:
    st.sidebar.warning("Please enter your Google Gemini API Key to begin.")


app_mode = st.sidebar.radio("Choose a Step", ["Step 1: Map Deployment Types", "Step 2: Map Metadata", "Step 3: Generate Keywords"])


if app_mode == "Step 1: Map Deployment Types":
    st.header("Step 1: Map Deployment Types")
    st.markdown("Upload a `.txt` file of URLs. This tool will scrape each URL for its deployment type, using an AI model for pages without clear tags. Download the resulting CSV to use in Step 2.")
    
    urls_file_step1 = st.file_uploader("Upload URLs File (.txt)", type="txt", key="step1_uploader")

    if st.button("🚀 Map Deployment Types", type="primary", disabled=(not GOOGLE_API_KEY)):
        if urls_file_step1 and GOOGLE_API_KEY:
            urls = [line.strip() for line in io.StringIO(urls_file_step1.getvalue().decode("utf-8")) if line.strip()]
            
            results, progress_bar = [], st.progress(0, "Starting...")
            for i, url in enumerate(urls):
                progress_bar.progress((i + 1) / len(urls), f"Processing URL {i+1}/{len(urls)}")
                soup, title = analyze_page_content(url)
                if soup:
                    dtype = get_deployment_type_from_scraping(soup) or get_deployment_type_with_llm(model, soup)
                    results.append({'Page Title': title, 'Page URL': url, 'Deployment Type': dtype})
                else:
                    results.append({'Page Title': title, 'Page URL': url, 'Deployment Type': 'Fetch Error'})
                time.sleep(1.1) # Rate limit to avoid overwhelming the API
            
            st.session_state.report_df_step1 = pd.DataFrame(results)
            st.success("✅ Step 1 complete! You can now download the report.")

    if 'report_df_step1' in st.session_state:
        st.subheader("Results")
        st.dataframe(st.session_state.report_df_step1)
        csv_data = st.session_state.report_df_step1.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Deployment Report (CSV)", csv_data, "deployment_report.csv", "text/csv")

elif app_mode == "Step 2: Map Metadata":
    st.header("Step 2: Map Metadata")
    st.markdown("Upload the CSV from Step 1, along with `.txt` files for topics, functional areas, and user roles. The AI will analyze each URL's content to map the most relevant metadata.")

    csv_file_step2 = st.file_uploader("Upload Deployment Report (.csv)", type="csv", key="step2_csv_uploader")
    topics_file = st.file_uploader("Upload Topics File (.txt)", type="txt", key="step2_topics")
    areas_file = st.file_uploader("Upload Functional Areas File (.txt)", type="txt", key="step2_areas")
    roles_file = st.file_uploader("Upload User Roles File (.txt)", type="txt", key="step2_roles")

    if st.button("🚀 Map Metadata", type="primary", disabled=(not GOOGLE_API_KEY)):
        if all([csv_file_step2, topics_file, areas_file, roles_file, GOOGLE_API_KEY]):
            df = pd.read_csv(csv_file_step2)
            
            topics = [line.strip() for line in io.StringIO(topics_file.getvalue().decode("utf-8")) if line.strip()]
            areas = [line.strip() for line in io.StringIO(areas_file.getvalue().decode("utf-8")) if line.strip()]
            roles = [line.strip() for line in io.StringIO(roles_file.getvalue().decode("utf-8")) if line.strip()]

            analysis_results = []
            progress_bar = st.progress(0, "Starting metadata mapping...")
            for i, row in df.iterrows():
                progress_bar.progress((i + 1) / len(df), f"Analyzing URL {i+1}/{len(df)}")
                soup, _ = analyze_page_content(row['Page URL'])
                if soup:
                    llm_data = get_metadata_analysis_with_llm(model, soup, roles, areas, topics)
                    error_msg = llm_data.get('error', 'Unknown LLM Error')
                    row['User Role'] = llm_data.get('user_role', error_msg)
                    row['Functional Area'] = llm_data.get('functional_area', error_msg)
                    row['Topics'] = llm_data.get('topics', error_msg)
                else:
                    row['User Role'], row['Functional Area'], row['Topics'] = 'Fetch Error', 'Fetch Error', 'Fetch Error'
                analysis_results.append(row)
                time.sleep(1.1) # Rate limit
            
            st.session_state.report_df_step2 = pd.DataFrame(analysis_results)
            st.success("✅ Step 2 complete! You can now download the metadata report.")
        else:
            st.warning("⚠️ Please upload all required files.")

    if 'report_df_step2' in st.session_state:
        st.subheader("Metadata Report")
        st.dataframe(st.session_state.report_df_step2)
        csv_data = st.session_state.report_df_step2.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Metadata Report (CSV)", csv_data, "metadata_report.csv", "text/csv")

elif app_mode == "Step 3: Generate Keywords":
    st.header("Step 3: Generate Keywords")
    st.markdown("Upload the CSV from Step 2. The AI will analyze each URL's content to generate 20 unique keywords.")

    csv_file_step3 = st.file_uploader("Upload Metadata Report (.csv)", type="csv", key="step3_csv_uploader")
    
    if st.button("🚀 Generate Keywords", type="primary", disabled=(not GOOGLE_API_KEY)):
        if csv_file_step3 and GOOGLE_API_KEY:
            df = pd.read_csv(csv_file_step3)

            analysis_results = []
            progress_bar = st.progress(0, "Starting keyword generation...")
            for i, row in df.iterrows():
                progress_bar.progress((i + 1) / len(df), f"Analyzing URL {i+1}/{len(df)}")
                soup, title = analyze_page_content(row['Page URL'])
                if soup:
                    llm_data = get_keywords_with_llm(model, soup, title)
                    error_msg = llm_data.get('error', 'Unknown LLM Error')
                    row['Keywords'] = llm_data.get('keywords', error_msg)
                else:
                    row['Keywords'] = 'Fetch Error'
                analysis_results.append(row)
                time.sleep(1.1) # Rate limit
            
            st.session_state.report_df_step3 = pd.DataFrame(analysis_results)
            st.success("✅ Step 3 complete! You can now download the final report.")
        else:
            st.warning("⚠️ Please upload the metadata report from Step 2.")
    
    if 'report_df_step3' in st.session_state:
        st.subheader("Final Report with Keywords")
        st.dataframe(st.session_state.report_df_step3)
        csv_data = st.session_state.report_df_step3.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Final Report (CSV)", csv_data, "final_report_with_keywords.csv", "text/csv")
