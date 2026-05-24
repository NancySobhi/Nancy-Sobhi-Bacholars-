import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
import tempfile, os, pandas as pd, json, re, io, xlsxwriter

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Financial Intelligence Framework", layout="wide")
st.title("Financial Intelligence Framework 📊")

# Initialize LLM (Ollama Llama 3.2)
llm = OllamaLLM(model="llama3.2", temperature=0)

# --- 2. PERSISTENT SESSION STATE ---
# This ensures conversation history and outputs do not disappear
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. CORE ARCHITECTURAL COMPONENTS ---

def classify_question(query):
    """STRICT IF-STATEMENT BASED CLASSIFICATION using keywords."""
    analytical_keywords = [
        'compare', 'comparison', 'trend', 'analyze', 'analysis', 'calculate', 
        'percentage', 'growth', 'dashboard', 'graph', 'chart', 'visualize', 
        'ranking', 'rank', 'highest', 'lowest', 'average', 'summarize', 
        'grouped', 'aggregation', 'sql', 'query', 'across periods', 'over time', 
        'variance', 'difference', 'correlation', 'insights', 'forecast', 
        'breakdown', 'categories', 'segment', 'distribution', 'performance', 
        'top', 'bottom'
    ]
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in analytical_keywords):
        return "ANALYTICAL"
    return "STANDARD"

def standard_agent(context, query):
    """STANDARD QUESTION LOGIC: Direct RAG Retrieval."""
    prompt = f"""
    You are a financial assistant. Use the PDF context to answer.
    CONTEXT: {context[:20000]}
    QUESTION: {query}
    
    RULE: If the answer is not in the PDF, return EXACTLY: 
    "Not in the PDF — it's not mentioned in the financial statement."
    """
    return llm.invoke(prompt)

def extraction_agent(context, query):
    """ANALYTICAL EXTRACTION: Structures data into JSON."""
    prompt = f"""
    Context: {context[:25000]}
    Task: Extract financial data for: {query}
    
    RULES:
    1. If data is missing, return EXACTLY: [NOT_FOUND]
    2. Format Periods as: 'Month Day Year' (e.g., March 29 2025).
    3. Use descriptive column names (e.g., 'Total Revenue', 'Net Sales').
    4. Output ONLY a valid JSON list of objects.
    
    EXAMPLE FORMAT:
    [
      {{"Period": "December 31 2024", "Net Income": 50000}},
      {{"Period": "December 31 2023", "Net Income": 45000}}
    ]
    """
    return llm.invoke(prompt)

def validator_agent(context, df):
    """VALIDATION AGENT: Compares extracted data against source."""
    if df is None or df.empty:
        return "FAIL"
    
    data_summary = df.to_string()
    prompt = f"""
    Compare the following extracted data against the PDF context.
    CONTEXT: {context[:10000]}
    EXTRACTED_DATA: {data_summary}
    
    TASK: Verify every number exists in the PDF.
    If accurate, return 'PASS'. If any value is hallucinated or modified, return 'FAIL'.
    """
    response = llm.invoke(prompt).upper()
    return "PASS" if "PASS" in response else "FAIL"

def dashboard_generator(df):
    """DASHBOARD REQUIREMENTS: Generates charts and Excel files."""
    excel_out = io.BytesIO()
    csv_out = io.BytesIO()
    
    with pd.ExcelWriter(excel_out, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Financial_Data')
        workbook = writer.book
        worksheet = workbook.add_worksheet('Dashboard')
        
        # Determine best chart type based on columns
        chart_type = 'column' # Default
        if len(df) > 3: chart_type = 'line' # Trend
        
        chart = workbook.add_chart({'type': chart_type})
        for i in range(1, len(df.columns)):
            chart.add_series({
                'name':       ['Financial_Data', 0, i],
                'categories': ['Financial_Data', 1, 0, len(df), 0],
                'values':     ['Financial_Data', 1, i, len(df), i],
            })
        chart.set_title({'name': f'Analytical Report: {df.columns[1]}'})
        worksheet.insert_chart('B2', chart)
        
    df.to_csv(csv_out, index=False)
    return excel_out.getvalue(), csv_out.getvalue()

# --- 4. FILE PROCESSING ---
uploaded_file = st.sidebar.file_uploader("Upload Financial PDF", type="pdf")
pdf_context = ""

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    pdf_context = "\n".join([p.page_content for p in pages])
    os.remove(tmp_path)
    st.sidebar.success("PDF Uploaded and Contextualized.")

# --- 5. CHAT UI & PERSISTENCE ---
# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "df" in msg:
            st.dataframe(msg["df"])
            st.download_button("📥 Download Excel Report", msg["excel"], "report.xlsx", key=f"ex_{msg['id']}")
            st.download_button("📥 Download CSV", msg["csv"], "data.csv", key=f"csv_{msg['id']}")

# User Input
if prompt := st.chat_input("Ask about your financial document..."):
    # Save User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Logic
    with st.chat_message("assistant"):
        if not pdf_context:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Analyzing..."):
                classification = classify_question(prompt)
                
                # --- IF STANDARD ---
                if classification == "STANDARD":
                    response = standard_agent(pdf_context, prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                # --- IF ANALYTICAL ---
                else:
                    raw_extraction = extraction_agent(pdf_context, prompt)
                    
                    if "[NOT_FOUND]" in raw_extraction:
                        msg = "Not in the PDF — it's not mentioned in the financial statement."
                        st.error(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                    else:
                        # Parsing JSON Logic
                        try:
                            json_match = re.search(r'(\[.*\])', raw_extraction, re.DOTALL)
                            if json_match:
                                data = json.loads(json_match.group(1).replace("'", '"'))
                                df = pd.DataFrame(data)
                                
                                # Numeric Cleanup
                                for col in df.columns:
                                    if col != "Period":
                                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

                                # VALIDATION STEP
                                validation_status = validator_agent(pdf_context, df)
                                
                                if validation_status == "PASS":
                                    excel_bin, csv_bin = dashboard_generator(df)
                                    st.success("Analysis Validated.")
                                    st.dataframe(df)
                                    
                                    # Save to Session State
                                    msg_id = len(st.session_state.messages)
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": "Analytical processing complete. Dashboard and files generated.",
                                        "df": df, "excel": excel_bin, "csv": csv_bin, "id": msg_id
                                    })
                                    # Trigger rerun to show download buttons immediately
                                    st.rerun()
                                else:
                                    msg = "Validation failed. Please reupload the document."
                                    st.error(msg)
                                    st.session_state.messages.append({"role": "assistant", "content": msg})
                            else:
                                raise ValueError("JSON missing")
                        except Exception as e:
                            msg = "Extraction failed. Please reupload the document."
                            st.error(msg)
                            st.session_state.messages.append({"role": "assistant", "content": msg})