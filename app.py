import streamlit as st
import pandas as pd
import os
import zipfile
from agent import run_agent

# --- Smart Dataset Loader Function ---
def load_demo_dataset():
    """Smart dataset loader that handles both zip and CSV files for GitHub deployment"""
    # Try zip file first (for GitHub deployment)
    zip_path = "GSE68086_TEP_data_matrix.csv.zip"
    csv_path = "GSE68086_TEP_data_matrix.csv"
    extracted_path = "GSE68086_TEP_data_matrix_extracted.csv"
    
    # Check if we already have an extracted version
    if os.path.exists(extracted_path):
        return extracted_path
        
    # Try to extract from ZIP file
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract the CSV file
                zip_ref.extractall(".")
                # The extracted file should be GSE68086_TEP_data_matrix.csv
                if os.path.exists(csv_path):
                    # Rename to avoid conflicts and mark as extracted
                    os.rename(csv_path, extracted_path)
                    return extracted_path
        except Exception as e:
            st.error(f"Error extracting dataset: {e}")
            return None
    
    # Fallback: try direct CSV file (for local development)
    elif os.path.exists(csv_path):
        return csv_path
        
    # No dataset found
    return None

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="BioScale Agent",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling (Tesla Project Style Guide) ---
st.markdown("""
<style>
    /* Import system fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Color Palette from Style Guide */
    :root {
        --primary-bg: #f4f3f1;
        --primary-fg: #022029;
        --secondary-bg: #022029;
        --secondary-fg: #f4f3ef;
        --neutral-bg: #ffffff;
        --neutral-fg: #222222;
        --neutral-muted: #f1f5f9;
        --neutral-muted-fg: #64748b;
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
        --info: #3b82f6;
    }
    
    /* Global Styling */
    .stApp {
        background-color: var(--primary-bg);
        font-family: system-ui, -apple-system, sans-serif;
        color: var(--primary-fg);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, var(--secondary-bg) 0%, #034a5a 100%);
        color: var(--secondary-fg);
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.75rem;
        color: var(--secondary-fg);
    }
    
    .hero-subtitle {
        font-size: 1rem;
        font-weight: 400;
        line-height: 1.5;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: var(--neutral-bg);
        border-right: 1px solid #e5e7eb;
        width: 25%;
        position: sticky;
        top: 1rem;
    }
    
    .sidebar .element-container {
        background-color: var(--neutral-bg);
    }
    
    /* Card Styling - Following Style Guide */
    .metric-card {
        background: var(--neutral-bg);
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Button Styling - Following Style Guide */
    .stButton > button {
        background: var(--secondary-bg);
        color: var(--secondary-fg);
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-weight: 400;
        font-size: 1rem;
        line-height: 1.6;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: #033c4a;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    
    /* Chat Input Enhancement */
    .stChatInputContainer {
        background: var(--neutral-bg);
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Chat Interface */
    .stChatMessage {
        background: var(--neutral-bg);
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        padding: 1.5rem;
    }
    
    /* Typography Scale - Following Style Guide */
    h1 {
        color: var(--primary-fg);
        font-weight: 700;
        font-size: 3rem;
        line-height: 1.2;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: var(--primary-fg);
        font-weight: 700;
        font-size: 2rem;
        line-height: 1.3;
        margin-bottom: 1.5rem;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: var(--primary-fg);
        font-weight: 600;
        font-size: 1.25rem;
        line-height: 1.4;
        margin-bottom: 1rem;
    }
    
    h4 {
        color: var(--primary-fg);
        font-weight: 600;
        font-size: 1.125rem;
        line-height: 1.4;
        margin-bottom: 1rem;
    }
    
    /* Body text */
    p, .stMarkdown {
        font-size: 1rem;
        font-weight: 400;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Small text */
    .stCaption, small {
        font-size: 0.875rem;
        font-weight: 400;
        line-height: 1.5;
        color: var(--neutral-muted-fg);
    }
    
    /* Sections - Following Style Guide */
    .section {
        background: var(--primary-bg);
        color: var(--primary-fg);
        padding: 6rem 0 4rem 0;
        margin-bottom: 4rem;
    }
    
    /* Tables - Following Style Guide */
    .stDataFrame {
        background: var(--neutral-bg);
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stDataFrame thead {
        background: var(--secondary-bg);
        color: var(--secondary-fg);
    }
    
    .stDataFrame tbody tr:nth-child(even) {
        background: #f9fafb;
    }
    
    .stDataFrame tbody tr:nth-child(odd) {
        background: var(--neutral-bg);
    }
    
    .stDataFrame th, .stDataFrame td {
        padding: 0.75rem 1rem;
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background: var(--success);
        border: none;
        border-radius: 0.5rem;
        color: white;
    }
    
    .stInfo {
        background: var(--info);
        border: none;
        border-radius: 0.5rem;
        color: white;
    }
    
    .stWarning {
        background: var(--warning);
        border: none;
        border-radius: 0.5rem;
        color: white;
    }
    
    .stError {
        background: var(--error);
        border: none;
        border-radius: 0.5rem;
        color: white;
    }
    
    /* Metrics */
    .css-1xarl3l {
        background: var(--neutral-bg);
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* JSON Display */
    .stJson {
        background: var(--neutral-bg);
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Code blocks */
    .stCode {
        background: var(--neutral-muted);
        border-radius: 0.5rem;
        font-family: ui-monospace, monospace;
        padding: 0.75rem 1rem;
    }
    
    /* Progress bars */
    .stProgress .css-zt5igj {
        background: var(--secondary-bg);
    }
    
    /* File uploader */
    .css-1cpxqw2 {
        background: var(--neutral-bg);
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1.5rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--neutral-bg);
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Status Indicators */
    .stStatus {
        background: var(--neutral-bg);
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Lists - Following Style Guide */
    ul {
        padding-left: 1.5rem;
        margin: 1rem 0;
    }
    
    li {
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    
    /* Focus states */
    *:focus {
        outline: 2px solid var(--info);
        outline-offset: 2px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design - Following Style Guide */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.75rem;
        }
        .hero-subtitle {
            font-size: 0.875rem;
        }
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .metric-card {
            margin-bottom: 1rem;
        }
        .stChatMessage {
            margin-bottom: 0.5rem;
        }
        .section {
            padding: 3rem 0 2rem 0;
        }
        h1 { font-size: 2rem; }
        h2 { font-size: 1.5rem; }
    }
    
    @media (min-width: 768px) and (max-width: 1024px) {
        .hero-title {
            font-size: 2rem;
        }
        .hero-subtitle {
            font-size: 0.95rem;
        }
        .section {
            padding: 4rem 0 3rem 0;
        }
    }
    
    @media (min-width: 1280px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .hero-subtitle {
            font-size: 1rem;
        }
        .section {
            padding: 6rem 0 4rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
# This ensures that the chat history and data path persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_path" not in st.session_state:
    # Set default dataset for demo using smart loader
    default_data_path = load_demo_dataset()
    if default_data_path and os.path.exists(default_data_path):
        st.session_state.data_path = default_data_path
    else:
        st.session_state.data_path = None

# --- Hero Section ---
st.markdown("""
<div class="hero-section">
    <div class="hero-title">BioScale</div>
    <div class="hero-subtitle">
        A Hybrid Biomedical AI Agent for Gene Expression Analysis
        <br><br>
        Professional-grade biomedical data analysis powered by intelligent routing between verified tools and dynamic AI code generation.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Demo Features Section ---
st.markdown("## Platform Capabilities")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: var(--secondary-bg); margin-bottom: 0.5rem;">📊 Dataset</h3>
        <p style="font-size: 0.875rem; color: var(--neutral-muted-fg); margin: 0;">GSE68086 Multi-Cancer<br>57,736 genes × 285 samples</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: var(--secondary-bg); margin-bottom: 0.5rem;">🛠️ Tools</h3>
        <p style="font-size: 0.875rem; color: var(--neutral-muted-fg); margin: 0;">Verified Functions<br>Reliable analysis methods</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: var(--secondary-bg); margin-bottom: 0.5rem;">🤖 AI Engine</h3>
        <p style="font-size: 0.875rem; color: var(--neutral-muted-fg); margin: 0;">Dynamic Code Generation<br>Novel query handling</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: var(--secondary-bg); margin-bottom: 0.5rem;">📈 Visualization</h3>
        <p style="font-size: 0.875rem; color: var(--neutral-muted-fg); margin: 0;">Interactive Plotly<br>Publication-ready plots</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Sidebar for Data Management ---
with st.sidebar:
    st.markdown("### Dataset Management")
    
    # Show current dataset status
    if st.session_state.data_path:
        if "GSE68086_TEP_data_matrix.csv" in st.session_state.data_path:
            st.success("**Demo Dataset Active**")
            st.info("""
            **GSE68086 Multi-Cancer Dataset**
            
            **Scale**: 57,736 genes × 285 samples  
            **Types**: Breast, GBM, Lung, CRC, Pancreatic  
            **Controls**: Healthy tissue samples included  
            **Format**: Gene expression matrix (CSV)
            """)
        else:
            st.success("**Custom Dataset Loaded**")
            file_name = os.path.basename(st.session_state.data_path)
            st.info(f"**File**: {file_name}")
    
    st.markdown("---")
    
    # Custom data upload section
    st.markdown("#### Upload Custom Data")
    st.markdown("*Optional: Upload your own gene expression matrix*")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="Upload your own gene expression data to analyze with BioScale"
    )

    if uploaded_file is not None:
        # Create a temporary directory to store uploaded files
        data_dir = "temp_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Save the file to get a stable path that the agent can access
        st.session_state.data_path = os.path.join(data_dir, uploaded_file.name)
        with open(st.session_state.data_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"**{uploaded_file.name}** uploaded successfully!")

    st.markdown("---")

    # Data Preview
    if st.session_state.data_path:
        st.markdown("#### Data Preview")
        try:
            df = pd.read_csv(st.session_state.data_path, nrows=5)
            st.dataframe(df, height=200, use_container_width=True)
            
            # Show basic stats
            st.markdown(f"""
            **Quick Stats:**
            - **Rows**: {len(df)} (preview)
            - **Columns**: {len(df.columns)}
            """)
        except Exception as e:
            st.error(f"Preview error: {e}")
    
    st.markdown("---")
    
    # Reset to demo data
    if st.button("**Reset to Demo Dataset**", use_container_width=True):
        default_data_path = load_demo_dataset()
        if default_data_path and os.path.exists(default_data_path):
            st.session_state.data_path = default_data_path
            st.rerun()
    
    st.markdown("---")
    
    # Technical Information
    st.markdown("#### Technical Details")
    st.markdown("""
    **Analysis Methods:**
    - Verified statistical tools
    - Dynamic code generation  
    - Interactive Plotly visualizations
    - Smart query routing
    
    **Supported Analysis:**
    - Gene expression profiling
    - Differential expression
    - PCA & clustering
    - Pathway analysis
    - Statistical testing
    """, help="BioScale combines verified tools with AI-generated code for comprehensive analysis")

# --- Sample Questions Section ---
if st.session_state.data_path:
    st.markdown("## Quick Start Analysis")
    st.markdown("**Click any question below to see BioScale in action with instant analysis:**")
    st.markdown("<br>", unsafe_allow_html=True)
    
    sample_questions = [
        {
            "title": "Dataset Overview",
            "question": "Can you give me an overview of the dataset's structure?",
            "icon": "",
            "description": "Get comprehensive statistics and data structure insights"
        },
        {
            "title": "Variable Genes Heatmap", 
            "question": "Generate a heatmap of the top 10 most variable genes across all samples.",
            "icon": "",
            "description": "Visualize the most variable genes across all samples"
        },
        {
            "title": "PCA Sample Relationships",
            "question": "Create a PCA plot to visualize the relationship between the samples.",
            "icon": "", 
            "description": "Explore sample clustering and cancer type relationships"
        },
        {
            "title": "Top Expression Analysis",
            "question": "Find the top 5 genes with the highest average expression across all samples and list their mean values.",
            "icon": "",
            "description": "Identify the most highly expressed genes in the dataset"
        },
        {
            "title": "Differential Expression",
            "question": "Create a volcano plot to show differentially expressed genes between Glioblastoma and Breast cancer samples.",
            "icon": "",
            "description": "Compare gene expression between cancer types"
        }
    ]
    
    # Create columns for sample questions
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        with cols[i % 2]:
            # Create styled button
            if st.button(
                f"**{q['title']}**\n\n{q['description']}", 
                key=f"sample_q_{i}", 
                use_container_width=True,
                help=q['question']
            ):
                # Add the question to chat and process it
                st.session_state.messages.append({"role": "user", "type": "text", "content": q['question']})
                
                # Process the question
                with st.spinner("BioScale agent is analyzing..."):
                    response = run_agent(q['question'], st.session_state.data_path)
                    st.session_state.messages.append({"role": "assistant", **response})
                
                st.rerun()
    
    st.markdown("---", unsafe_allow_html=True)

# --- Main Chat Interface ---
st.markdown("## Interactive Analysis Chat")
st.markdown("**Ask questions about your data in natural language and get instant analysis:**")

# Chat input
if st.session_state.data_path is None:
    st.warning("**No dataset available.** Please upload a custom file or reset to the demo dataset using the sidebar.")
    prompt = st.chat_input("Please load a dataset first to start analyzing...", disabled=True)
else:
    prompt = st.chat_input("Ask anything about your biomedical data... (e.g., 'Compare gene expression between cancer types')")

# Handle user input
if prompt:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    
    # Show processing status
    with st.status("**BioScale AI Agent Processing**", expanded=True) as status:
        st.write("**Loading dataset...**")
        st.write("**Analyzing your question...**")
        st.write("**Choosing optimal analysis method...**")
        st.write("**Executing analysis...**")
        st.write("**Generating visualizations...**")
        
        try:
            response = run_agent(prompt, st.session_state.data_path)
            st.session_state.messages.append({"role": "assistant", **response})
            status.update(label="**Analysis Complete!**", state="complete", expanded=False)
        except Exception as e:
            st.error(f"**Analysis failed:** {str(e)}")
            status.update(label="**Analysis Failed**", state="error", expanded=False)
    
    st.rerun()

st.markdown("---")

# Display chat messages from history on app rerun (newest first)
if st.session_state.messages:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### **Chat History** (Most Recent First)")
    with col2:
        if st.button("**Clear Chat**", key="clear_chat", help="Clear all chat history"):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown(f"**{len(st.session_state.messages)} total messages** • Latest responses shown first")
    st.markdown("---")
    
    # Reverse the order to show newest messages first
    for idx, message in enumerate(reversed(st.session_state.messages)):
        # Add a subtle separator between messages
        if idx > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            
        with st.chat_message(message["role"]):
            # Add timestamp info for context
            if message["role"] == "user":
                st.markdown(f"**Your Question:**")
            else:
                st.markdown(f"**BioScale Analysis:**")
            
            # Check the type of content to display
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "plot":
                st.markdown(message["content"])
                # Check if image file exists before trying to display it
                if "path" in message and os.path.exists(message["path"]):
                    st.image(message["path"])
                else:
                    st.warning("Plot image file not found. The analysis may have generated an HTML plot instead.")
            elif message["type"] == "plotly":
                st.markdown(message["content"])
                st.components.v1.html(message["html"], height=600)
            elif message["type"] == "json":
                st.markdown(f"**{message['content']}**")
                
                # Create a nice display for different types of data
                data = message["data"]
                if "details" in data:
                    details = data["details"]
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Genes", details.get("genes", "N/A"))
                        st.metric("Total Samples", details.get("samples", "N/A"))
                    
                    with col2:
                        if "gene_id_preview" in details:
                            st.write("**Gene ID Preview:**")
                            for gene_id in details["gene_id_preview"]:
                                st.code(gene_id)
                        
                        if "sample_preview" in details:
                            st.write("**Sample Preview:**")
                            for sample in details["sample_preview"]:
                                st.code(sample)
                
                # Show expandable full JSON for advanced users
                with st.expander("View Full JSON Output"):
                    st.json(data)
            elif message["type"] == "structured_data":
                st.markdown(f"**{message['content']}**")
                
                data = message["data"]
                if isinstance(data, dict):
                    # Display dictionary data nicely
                    if len(data) <= 10:  # Show small dicts as metrics/key-value pairs
                        cols = st.columns(min(len(data), 3))
                        for idx_inner, (key, value) in enumerate(data.items()):
                            with cols[idx_inner % 3]:
                                if isinstance(value, (int, float)):
                                    st.metric(str(key).replace("_", " ").title(), f"{value:.4f}" if isinstance(value, float) else value)
                                else:
                                    st.write(f"**{str(key).replace('_', ' ').title()}:**")
                                    st.code(str(value))
                    else:
                        # Large dict - show as expandable JSON
                        st.json(data)
                        
                elif isinstance(data, list):
                    # Display list data as a table or items
                    if len(data) > 0 and isinstance(data[0], (str, int, float)):
                        # Simple list - show as table
                        df = pd.DataFrame({"Value": data})
                        st.dataframe(df, use_container_width=True)
                    else:
                        # Complex list - show as JSON
                        st.json(data)
                else:
                    # Fallback
                    st.json(data)
            elif message["type"] == "error":
                st.error(message["content"])
else:
    # Show helpful message when no chat history exists
    st.markdown("### **Start Your Analysis**")
    st.info("**Use the chat box below to ask questions about your biomedical data** or try one of the sample questions to get started!")
    
    # Show some example questions as inspiration
    st.markdown("**Example Questions:**")
    examples = [
        "What are the top 10 most variable genes in this dataset?",
        "Show me a heatmap of the correlation between samples",
        "Create a volcano plot comparing two cancer types",
        "Perform differential expression analysis",
        "Generate a PCA plot showing sample relationships"
    ]
    for example in examples:
        st.markdown(f"• *{example}*")
