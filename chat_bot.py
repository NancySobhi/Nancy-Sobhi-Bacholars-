import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
import pandas as pd
import json
import re

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="Financial Thesis RAG", layout="wide")
st.title("Financial RAG Processor 📊")
llm = OllamaLLM(model="llama3.2", temperature=0) # Temp 0 is vital for financial accuracy

# --- 2. DATA VALIDATION & FORMATTING ENGINE ---
def format_and_verify_dataframe(df):
    """
    Forces headers to 'Total Amount', sorts dates, 
    and formats to '29, March, 2025'.
    """
    # Force rename metrics to your specs
    rename_map = {
        "Metric": "Total Amount", "Value": "Total Amount", 
        "Net Sales": "Total Amount", "Amount": "Total Amount"
    }
    df = df.rename(columns=rename_map)
    
    if 'Period' in df.columns:
        # Prevent 'None' errors by dropping empty rows
        df = df.dropna(subset=['Period'])
        # Convert to datetime for proper mathematical sorting
        df['dt_temp'] = pd.to_datetime(df['Period'], errors='coerce')
        df = df.dropna(subset=['dt_temp']).sort_values(by='dt_temp')
        # Format as requested: 29, March, 2025
        df['Period'] = df['dt_temp'].dt.strftime('%d, %B, %Y')
        df = df.drop(columns=['dt_temp'])
    
    return df

# --- 3. PERSISTENT CHAT STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. THE DOCUMENT FEED ---
uploaded_file = st.sidebar.file_uploader("Upload Annual Report (PDF)", type="pdf")
context = ""
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    # Load document content once to avoid re-reading
    context = "\n".join([p.page_content for p in loader.load()])
    os.remove(tmp_path)

# Display Session History (Keeps scrolling possible)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "df" in msg:
            st.dataframe(msg["df"])
            csv = msg["df"].to_csv(index=False).encode('utf-8')
            st.download_button("Download Verified CSV", csv, "verified_finance.csv", "text/csv", key=f"dl_{msg['id']}")

# --- 5. THE DUAL-AGENT CORE ---
def route_query(query):
    # Differentiates between Standard Fact vs Analytical Data Extraction
    analytical_keywords = ['compare', 'versus', 'vs', 'across', 'all periods', 'trend', 'growth']
    if any(k in query.lower() for k in analytical_keywords):
        return "ANALYTICAL"
    return "STANDARD"

if prompt := st.chat_input("Ask a question about the report..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not context:
            st.error("Please upload the required PDF/s first.")
        else:
            q_type = route_query(prompt)
            
            if q_type == "STANDARD":
                # Standard Logic: Direct Retrieval
                sys_prompt = f"Context: {context[:8000]}\nQuestion: {prompt}\nRule: Answer ONLY from PDF. If not present, say 'This is not included in the PDF.'"
                response = llm.invoke(sys_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            else:
                # Analytical Logic: Extraction -> Validation Phase
                with st.spinner("Agent 1: Extracting structured data..."):
                    extract_prompt = f"Context: {context[:18000]}\nTask: Extract financial data for: {prompt}\nOutput: JSON list only with keys 'Period' and metrics."
                    raw_out = llm.invoke(extract_prompt)
                    
                    # Clean JSON string (remove markdown)
                    clean_json = re.search(r'\[.*\]', re.sub(r'```json|```', '', raw_out), re.DOTALL)
                    
                    if clean_json:
                        try:
                            data = json.loads(clean_json.group(0))
                            
                            # THE VALIDATION PHASE (Agent 2)
                            with st.spinner("Agent 2: Validating against source PDF..."):
                                check_prompt = f"Context: {context[:18000]}\nData to verify: {data}\nInstruction: Cross-check these numbers. Reply only 'VALID' or 'INVALID'."
                                status = llm.invoke(check_prompt)
                                
                                if "VALID" in status.upper():
                                    final_df = format_and_verify_dataframe(pd.DataFrame(data))
                                    
                                    if not final_df.empty:
                                        st.markdown("### Verified Structured Data")
                                        st.dataframe(final_df)
                                        
                                        # Store unique ID for the download button key
                                        m_id = str(pd.Timestamp.now().timestamp())
                                        csv = final_df.to_csv(index=False).encode('utf-8')
                                        st.download_button("Download Verified CSV", csv, "verified_finance.csv", "text/csv", key=f"btn_{m_id}")
                                        
                                        st.session_state.messages.append({
                                            "role": "assistant", 
                                            "content": "Data extracted and verified successfully:", 
                                            "df": final_df, "id": m_id
                                        })
                                    else:
                                        st.error("This is not included in the PDF.")
                                else:
                                    # Termination logic as per your requirement
                                    st.error("Validation Failed: Extracted values do not match the PDF. Process terminated. Please verify your document.")
                        except Exception as e:
                            st.error("This is not included in the PDF.")
                    else:
                        st.error("This is not included in the PDF.")