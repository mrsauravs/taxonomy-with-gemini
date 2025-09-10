import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import re
import json
import time
from collections import Counter

# --- LLM and API Configuration ---
# Using a dictionary to manage different providers and their default models
LLM_PROVIDERS = {
    "Google Gemini": "gemini-1.5-flash-latest",
    "OpenAI": "gpt-3.5-turbo",
    "Hugging Face": "HuggingFaceH4/zephyr-7b-beta" 
}

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

# --- LLM API Abstraction Layer ---

def call_llm_with_retry(provider, api_key, model, prompt, max_tokens):
    """
    Calls the appropriate LLM API based on the selected provider with retry logic.
    Handles different clients and response formats.
    """
    for attempt in range(3):
        try:
            if provider == "Google Gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model_instance = genai.GenerativeModel(model)
                response = model_instance.generate_content(prompt)
                return response.text

            elif provider == "OpenAI":
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            elif provider == "Hugging Face":
                from huggingface_hub import InferenceClient
                client = InferenceClient(token=api_key)
                response = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

        except Exception as e:
            error_message = str(e)
            if attempt < 2:
                st.warning(f"API call with {provider} failed (attempt {attempt + 1}/3): {error_message}. Retrying...")
                time.sleep(5)
            else:
                st.error(f"All retries failed for {provider} model {model}: {error_message}")
    
    raise Exception("LLM API Error: All retries failed.")

# --- Basic (Non-LLM) Analysis Functions ---

def get_metadata_basic(soup, roles, areas, topics):
    """Performs basic metadata mapping by counting keyword occurrences."""
    if not soup: return {"user_role": "Fetch Error", "functional_area": "Fetch Error", "topics": "Fetch Error"}
    content_text = (soup.find('article') or soup.body).get_text().lower()
    
    def find_best_match(text, term_list):
        counts = {term: text.count(term.lower()) for term in term_list}
        return max(counts, key=counts.get) if any(counts.values()) else "N/A"

    return {
        "user_role": find_best_match(content_text, roles),
        "functional_area": find_best_match(content_text, areas),
        "topics": find_best_match(content_text, topics)
    }

def get_keywords_basic(soup):
    """Performs basic keyword extraction based on word frequency."""
    if not soup: return {"keywords": "Fetch Error"}
    content_text = (soup.find('article') or soup.body).get_text().lower()
    words = re.findall(r'\b[a-z-]{4,}\b', content_text)
    
    stop_words = set(['the', 'and', 'for', 'with', 'this', 'that', 'you', 'are', 'not', 'can', 'from', 'alation', 'data', 'file', 'page', 'use', 'see', 'also', 'click', 'select', 'will', 'then', 'your', 'have', 'been', 'which', 'what', 'where', 'when', 'who', 'why', 'how'])
    filtered_words = [word for word in words if word not in stop_words]
    
    top_20 = [word for word, count in Counter(filtered_words).most_common(20)]
    return {"keywords": ', '.join(top_20)}

# --- LLM-Powered Analysis Functions ---

def get_deployment_type_with_llm(provider, api_key, model, soup):
    """Uses an LLM to infer deployment type if scraping fails."""
    if not soup: return "Analysis Error"
    content_text = (soup.find('article') or soup.body).get_text(separator=' ', strip=True)[:10000]

    prompt = f"""Analyze the content and determine if it's for 'Alation Cloud Service', 'Customer Managed', or both. Respond with ONLY one of these three options and nothing else.
    Content: ---
    {content_text}
    ---
    The correct deployment type is:"""
    
    try:
        response = call_llm_with_retry(provider, api_key, model, prompt, max_tokens=20)
        cleaned_response = response.strip().replace('"', '')
        valid_responses = ["Alation Cloud Service", "Customer Managed", "Alation Cloud Service, Customer Managed"]
        return f"{cleaned_response} (AI Inferred)" if cleaned_response in valid_responses else "LLM Inference Failed"
    except Exception as e:
        return f"LLM API Error: {str(e)}"

def get_llm_analysis(provider, api_key, model, prompt, max_tokens):
    """Helper to call the LLM and parse the JSON response."""
    try:
        response_text = call_llm_with_retry(provider, api_key, model, prompt, max_tokens)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"error": f"Failed to parse LLM response: {response_text[:200]}..."}
    except Exception as e:
        return {"error": f"LLM API Error: {str(e)}"}

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("📄 Hybrid Content Analysis Workflow")

# --- Sidebar for LLM Configuration ---
with st.sidebar:
    st.header("🤖 LLM Configuration")
    selected_provider = st.selectbox("Choose LLM Provider", list(LLM_PROVIDERS.keys()))
    
    model_to_use = LLM_PROVIDERS[selected_provider]
    if selected_provider == "Hugging Face":
        model_to_use = st.text_input(
            "Enter Hugging Face Model ID",
            value=LLM_PROVIDERS["Hugging Face"],
            help="Use any model compatible with the chat completion API, e.g., 'deepseek-ai/deepseek-coder-6.7b-instruct'."
        )

    api_key = st.text_input(f"Enter your {selected_provider} API Key", type="password")
    
    st.info("You can run initial analysis without an API key. The key is only required for AI-powered steps.")
    st.markdown("---")
    st.markdown("""
    **Note on `ollama`:** This app uses hosted APIs for providers like Hugging Face. To use local models via `ollama`, a different application setup connecting to a local server would be required.
    """)

# --- Main App Logic ---
app_mode = st.radio(
    "Choose a Step in the Workflow",
    ["Step 1: Map Deployment Types", "Step 2: Map Metadata", "Step 3: Generate Keywords"],
    horizontal=True
)

if app_mode == "Step 1: Map Deployment Types":
    st.header("Step 1: Map Deployment Types")
    st.markdown("Upload a `.txt` file of URLs. The app will first map deployment types using HTML scraping. You can then optionally use an AI model to fill in any missing values.")
    
    urls_file = st.file_uploader("Upload URLs File (.txt)", type="txt", key="step1_uploader")

    if st.button("🚀 Run Initial Scraping", key="step1_scrape"):
        if urls_file:
            urls = [line.strip() for line in io.StringIO(urls_file.getvalue().decode("utf-8")) if line.strip()]
            results = []
            progress_bar = st.progress(0, "Starting scraping...")
            for i, url in enumerate(urls):
                progress_bar.progress((i + 1) / len(urls), f"Scraping URL {i+1}/{len(urls)}")
                soup, title = analyze_page_content(url)
                dtype = get_deployment_type_from_scraping(soup) if soup else "Fetch Error"
                results.append({'Page Title': title, 'Page URL': url, 'Deployment Type': dtype})
            
            st.session_state.report_df_step1 = pd.DataFrame(results)
            st.success("✅ Initial scraping complete!")
        else:
            st.warning("Please upload a URLs file.")

    if 'report_df_step1' in st.session_state:
        st.subheader("Scraping Results")
        df1 = st.session_state.report_df_step1
        st.dataframe(df1)
        
        missing_rows = df1[df1['Deployment Type'] == '']
        if not missing_rows.empty:
            st.warning(f"Found {len(missing_rows)} rows with no deployment type.")
            if st.button("🤖 Fill Blanks with AI", help="Uses the configured LLM to analyze pages where scraping failed."):
                if not api_key:
                    st.error(f"Please enter your {selected_provider} API Key in the sidebar to use this feature.")
                else:
                    progress_bar = st.progress(0, "Starting AI analysis for missing rows...")
                    for i, (index, row) in enumerate(missing_rows.iterrows()):
                        progress_bar.progress((i + 1) / len(missing_rows), f"Analyzing URL {i+1}/{len(missing_rows)} with AI")
                        soup, _ = analyze_page_content(row['Page URL'])
                        if soup:
                            llm_dtype = get_deployment_type_with_llm(selected_provider, api_key, model_to_use, soup)
                            df1.loc[index, 'Deployment Type'] = llm_dtype
                        time.sleep(1.1)
                    st.session_state.report_df_step1 = df1
                    st.success("✅ AI analysis complete! Table updated.")
                    st.rerun()
        
        csv_data = df1.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Report (CSV)", csv_data, "deployment_report_step1.csv", "text/csv")


elif app_mode == "Step 2: Map Metadata":
    st.header("Step 2: Map Metadata")
    st.markdown("Upload the report from Step 1 and your metadata files. You can run a basic analysis (no AI) or an advanced AI-powered analysis.")

    csv_file_step2 = st.file_uploader("Upload Deployment Report (.csv)", type="csv", key="step2_uploader")
    topics_file = st.file_uploader("Upload Topics File (.txt)", type="txt", key="step2_topics")
    areas_file = st.file_uploader("Upload Functional Areas File (.txt)", type="txt", key="step2_areas")
    roles_file = st.file_uploader("Upload User Roles File (.txt)", type="txt", key="step2_roles")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Run Basic Analysis (No AI)", key="step2_basic"):
            if all([csv_file_step2, topics_file, areas_file, roles_file]):
                df2 = pd.read_csv(csv_file_step2)
                topics = [line.strip() for line in io.StringIO(topics_file.getvalue().decode("utf-8")) if line.strip()]
                areas = [line.strip() for line in io.StringIO(areas_file.getvalue().decode("utf-8")) if line.strip()]
                roles = [line.strip() for line in io.StringIO(roles_file.getvalue().decode("utf-8")) if line.strip()]
                
                results = []
                progress_bar = st.progress(0, "Starting basic analysis...")
                for i, row in df2.iterrows():
                    progress_bar.progress((i + 1) / len(df2), f"Analyzing URL {i+1}/{len(df2)}")
                    soup, _ = analyze_page_content(row['Page URL'])
                    metadata = get_metadata_basic(soup, roles, areas, topics)
                    row['User Role'] = metadata['user_role']
                    row['Functional Area'] = metadata['functional_area']
                    row['Topics'] = metadata['topics']
                    results.append(row)

                st.session_state.report_df_step2 = pd.DataFrame(results)
                st.success("✅ Basic analysis complete!")
            else:
                st.warning("Please upload all required files.")

    with col2:
        if st.button("🤖 Run AI-Powered Analysis", key="step2_ai"):
             if not api_key:
                 st.error(f"Please enter your {selected_provider} API Key in the sidebar.")
             elif all([csv_file_step2, topics_file, areas_file, roles_file]):
                df2 = pd.read_csv(csv_file_step2)
                topics = [line.strip() for line in io.StringIO(topics_file.getvalue().decode("utf-8")) if line.strip()]
                areas = [line.strip() for line in io.StringIO(areas_file.getvalue().decode("utf-8")) if line.strip()]
                roles = [line.strip() for line in io.StringIO(roles_file.getvalue().decode("utf-8")) if line.strip()]

                results = []
                progress_bar = st.progress(0, "Starting AI analysis...")
                for i, row in df2.iterrows():
                    progress_bar.progress((i + 1) / len(df2), f"Analyzing URL {i+1}/{len(df2)}")
                    soup, _ = analyze_page_content(row['Page URL'])
                    if soup:
                        prompt = f"""You are an expert content analyst. Analyze the provided content. From the available lists, select the MOST RELEVANT User Role(s), Functional Area(s), and Topic(s).
                        Available User Roles: {', '.join(roles)}
                        Available Functional Areas: {', '.join(areas)}
                        Available Topics: {', '.join(topics)}
                        Content: --- {soup.get_text()[:10000]} ---
                        Provide your response in a single JSON object format like this example: {{"user_role": "Steward", "functional_area": "Data Quality", "topics": "Data Quality Checks"}}"""
                        llm_data = get_llm_analysis(selected_provider, api_key, model_to_use, prompt, 256)
                        error_msg = llm_data.get('error', 'LLM Error')
                        row['User Role'] = llm_data.get('user_role', error_msg)
                        row['Functional Area'] = llm_data.get('functional_area', error_msg)
                        row['Topics'] = llm_data.get('topics', error_msg)
                    else:
                        row['User Role'], row['Functional Area'], row['Topics'] = 'Fetch Error', 'Fetch Error', 'Fetch Error'
                    results.append(row)
                    time.sleep(1.1)

                st.session_state.report_df_step2 = pd.DataFrame(results)
                st.success("✅ AI-Powered analysis complete!")
             else:
                st.warning("Please upload all required files.")

    if 'report_df_step2' in st.session_state:
        st.subheader("Metadata Report")
        st.dataframe(st.session_state.report_df_step2)
        csv_data_2 = st.session_state.report_df_step2.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Report (CSV)", csv_data_2, "metadata_report_step2.csv", "text/csv")


elif app_mode == "Step 3: Generate Keywords":
    st.header("Step 3: Generate Keywords")
    st.markdown("Upload the report from Step 2. You can generate keywords using a basic frequency count or an advanced AI model.")
    
    csv_file_step3 = st.file_uploader("Upload Metadata Report (.csv)", type="csv", key="step3_uploader")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Run Basic Keyword Generation (No AI)", key="step3_basic"):
            if csv_file_step3:
                df3 = pd.read_csv(csv_file_step3)
                results = []
                progress_bar = st.progress(0, "Starting basic keyword generation...")
                for i, row in df3.iterrows():
                    progress_bar.progress((i + 1) / len(df3), f"Analyzing URL {i+1}/{len(df3)}")
                    soup, _ = analyze_page_content(row['Page URL'])
                    keywords_data = get_keywords_basic(soup)
                    row['Keywords'] = keywords_data['keywords']
                    results.append(row)
                
                st.session_state.report_df_step3 = pd.DataFrame(results)
                st.success("✅ Basic keyword generation complete!")
            else:
                st.warning("Please upload the metadata report.")

    with col2:
        if st.button("🤖 Run AI-Powered Keyword Generation", key="step3_ai"):
            if not api_key:
                st.error(f"Please enter your {selected_provider} API Key in the sidebar.")
            elif csv_file_step3:
                df3 = pd.read_csv(csv_file_step3)
                results = []
                progress_bar = st.progress(0, "Starting AI keyword generation...")
                for i, row in df3.iterrows():
                    progress_bar.progress((i + 1) / len(df3), f"Analyzing URL {i+1}/{len(df3)}")
                    soup, title = analyze_page_content(row['Page URL'])
                    if soup:
                        connector_instructions = ""
                        if "OCF Connector" in title:
                            connector_name = title.split('|')[0].strip()
                            db_system = connector_name.replace("OCF Connector", "").strip()
                            connector_instructions = f'CRITICAL RULE: The keywords MUST include both "{connector_name}" and "{db_system} data source".'

                        prompt = f"""Generate exactly 20 unique, comma-separated technical keywords from the content.
                        EXCLUSION RULES: Exclude generic words like 'guide', 'documentation', 'button', 'click', 'data', 'alation', 'prerequisites', 'overview', 'steps'.
                        {connector_instructions}
                        Content: --- {soup.get_text()[:10000]} ---
                        Provide your response in a single JSON object format like this example: {{"keywords": "keyword1, keyword2, ..., keyword20"}}"""
                        llm_data = get_llm_analysis(selected_provider, api_key, model_to_use, prompt, 512)
                        row['Keywords'] = llm_data.get('keywords', llm_data.get('error', 'LLM Error'))
                    else:
                        row['Keywords'] = 'Fetch Error'
                    results.append(row)
                    time.sleep(1.1)

                st.session_state.report_df_step3 = pd.DataFrame(results)
                st.success("✅ AI-Powered keyword generation complete!")
            else:
                st.warning("Please upload the metadata report.")

    if 'report_df_step3' in st.session_state:
        st.subheader("Final Report with Keywords")
        st.dataframe(st.session_state.report_df_step3)
        csv_data_3 = st.session_state.report_df_step3.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Final Report (CSV)", csv_data_3, "final_report_step3.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© All rights reserved, Saurabh Sugandh</div>", unsafe_allow_html=True)
