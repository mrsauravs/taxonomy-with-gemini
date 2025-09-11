import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import re

# --- Utility and Scraping Functions ---

@st.cache_data
def analyze_page_content(url):
    """Fetches and parses a URL for its title and HTML content."""
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
    """Determines deployment type from parsed HTML."""
    if not soup: return ""
    if soup.find('p', class_='cloud-label') and soup.find('p', class_='on-prem-label'):
        return "Alation Cloud Service, Customer Managed"
    if soup.find('p', class_='cloud-label'): return "Alation Cloud Service"
    if soup.find('p', class_='on-prem-label'): return "Customer Managed"
    return "" # Return blank if no tag is found

def extract_main_content(soup):
    """Extracts main content by cleaning out boilerplate elements."""
    if not soup:
        return "Content Not Available"
    main_content = soup.find('article') or soup.find('main') or soup.body
    if main_content:
        for element in main_content.find_all(['nav', 'header', 'footer', 'aside']):
            element.decompose()
        return main_content.get_text(separator=' ', strip=True)
    return "Main Content Not Found"

# --- Mapping Helper Functions ---

def is_standalone_word(text, match):
    """Checks if a regex match is a standalone word."""
    start, end = match.start(), match.end()
    is_start_ok = (start == 0) or (text[start - 1].isspace() or text[start - 1] in '(),."\'')
    is_end_ok = (end == len(text)) or (text[end].isspace() or text[end] in '(),."\'')
    return is_start_ok and is_end_ok

def find_items_in_text(text, items):
    """Finds which items (roles, topics) from a list are present in the text."""
    if not isinstance(text, str): return ""
    found_items = []
    for item in items:
        for match in re.finditer(r'\b' + re.escape(item) + r'\b', text, re.IGNORECASE):
            if is_standalone_word(text, match):
                found_items.append(item)
                break
    return ", ".join(found_items) if found_items else "" # Return blank if no items are found

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📄 Web Content Mapper")
st.markdown("A three-step tool to map deployment types, user roles, and identify relevant topics.")

with st.expander("Step 1: Map Deployment Type", expanded=True):
    urls_file_step1 = st.file_uploader("Upload URLs File (.txt)", key="step1")
    if st.button("🚀 Scrape URLs", type="primary"):
        if urls_file_step1:
            urls = [line.strip() for line in io.StringIO(urls_file_step1.getvalue().decode("utf-8")) if line.strip()]
            results, pb = [], st.progress(0, "Starting...")
            for i, url in enumerate(urls):
                pb.progress((i + 1) / len(urls), f"Processing URL {i+1}/{len(urls)}...")
                soup, title = analyze_page_content(url)
                data = {'Page Title': title, 'Page URL': url}
                if soup:
                    data.update({
                        'Deployment Type': get_deployment_type_from_scraping(soup),
                        'Page Content': extract_main_content(soup)
                    })
                else:
                    data.update({'Deployment Type': 'Fetch Error', 'Page Content': 'Fetch Error'})
                results.append(data)
            
            st.session_state.df1 = pd.DataFrame(results)
            for key in ['df2', 'df3']:
                if key in st.session_state: del st.session_state[key]
            st.success("✅ Step 1 complete! Proceed to Step 2.")
        else:
            st.warning("⚠️ Please upload a URLs file.")

if 'df1' in st.session_state:
    with st.expander("Step 2: Map User Roles", expanded=True):
        roles_file = st.file_uploader("Upload User Roles File (.txt)", key="step2")
        if st.button("🗺️ Map User Roles"):
            if roles_file:
                roles = [line.strip() for line in io.StringIO(roles_file.getvalue().decode("utf-8")) if line.strip()]
                if roles:
                    df = st.session_state.df1.copy()
                    df['User Roles'] = df['Page Content'].apply(lambda txt: find_items_in_text(txt, roles))
                    st.session_state.df2 = df
                    if 'df3' in st.session_state: del st.session_state['df3']
                    st.success("✅ Step 2 complete! Proceed to Step 3.")
                else: st.warning("⚠️ Roles file is empty.")
            else: st.warning("⚠️ Please upload a roles file.")

if 'df2' in st.session_state:
    with st.expander("Step 3: Map Topics", expanded=True):
        topics_file = st.file_uploader("Upload Topics File (.txt)", key="step3")
        if st.button("🏷️ Map Topics"):
            if topics_file:
                topics = [line.strip() for line in io.StringIO(topics_file.getvalue().decode("utf-8")) if line.strip()]
                if topics:
                    df = st.session_state.df2.copy()
                    df['Topics'] = df['Page Content'].apply(lambda txt: find_items_in_text(txt, topics))
                    st.session_state.df3 = df
                    st.success("✅ Mapping complete! The final report is ready below.")
                else: st.warning("⚠️ Topics file is empty.")
            else: st.warning("⚠️ Please upload a topics file.")

st.markdown("---")
st.subheader("📊 Results")

df_to_display = pd.DataFrame()
if 'df3' in st.session_state:
    df_to_display = st.session_state.df3
elif 'df2' in st.session_state:
    df_to_display = st.session_state.df2
elif 'df1' in st.session_state:
    df_to_display = st.session_state.df1

if not df_to_display.empty:
    # Define the final columns for display and download
    final_columns = ['Page Title', 'Page URL', 'Deployment Type', 'User Roles', 'Topics']
    # Filter the DataFrame to only include columns that already exist
    display_columns = [col for col in final_columns if col in df_to_display.columns]
    
    st.dataframe(df_to_display[display_columns])
    csv_data = df_to_display[display_columns].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 Download Report (CSV)",
        data=csv_data,
        file_name="content_mapping_report.csv",
        mime="text/csv"
    )
else:
    st.write("Upload a file in Step 1 and click 'Scrape URLs' to generate a report.")
