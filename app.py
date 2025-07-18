import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import json
import zipfile
from agent import run_agent

# --- CRITICAL: Session State Initialization (must be first) ---
# Initialize session state variables early to prevent AttributeError
def init_session_state():
    """Initialize session state variables safely"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "data_path" not in st.session_state:
        st.session_state.data_path = None
    if "app_initialized" not in st.session_state:
        st.session_state.app_initialized = False
    if "user_choice" not in st.session_state:
        st.session_state.user_choice = None  # Will be "protocol" or "data_analysis"
    if "show_questions" not in st.session_state:
        st.session_state.show_questions = False

# Call initialization immediately
init_session_state()

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

# --- Protocol Display Function ---
def display_protocol(protocol_json: str):
    """Renders a protocol JSON object in a user-friendly format."""
    try:
        if '```json' in protocol_json:
            protocol_json = protocol_json.split('```json')[1].split('```')[0].strip()
        protocol = json.loads(protocol_json)
    except (json.JSONDecodeError, IndexError) as e:
        st.error(f"Failed to decode the protocol format: {e}")
        st.code(protocol_json)
        return

    st.subheader(protocol.get("title", "Untitled Protocol"))
    st.caption(f"Objective: {protocol.get('objective', 'Not specified')}")

    with st.expander("Materials and Equipment", expanded=True):
        materials = protocol.get("materials", {})
        if materials:
            for category, items in materials.items():
                st.markdown(f"**{category}:** {items}")
        else:
            st.markdown("No materials listed.")
    st.divider()

    st.subheader("Procedure")
    steps = protocol.get("steps", [])
    if not steps:
        st.markdown("No steps provided.")
        return

    for step in sorted(steps, key=lambda x: x.get('step_number', 0)):
        step_title = f"**Step {step['step_number']}**"
        if step.get('duration_minutes', 0) > 0:
            step_title += f" *({step['duration_minutes']} min)*"
        st.markdown(step_title)
        st.markdown(f"> {step['description']}")

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

# --- Dataset Initialization ---
# Load demo dataset if not already loaded
if not st.session_state.app_initialized:
    try:
        default_data_path = load_demo_dataset()
        if default_data_path and os.path.exists(default_data_path):
            st.session_state.data_path = default_data_path
        else:
            st.session_state.data_path = None
        st.session_state.app_initialized = True
    except Exception as e:
        st.error(f"Error loading demo dataset: {e}")
        st.session_state.data_path = None
        st.session_state.app_initialized = True

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

# Responsive layout for capability boxes
col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

with col1:
    st.markdown("""
    <div class="metric-card" style="min-height: 120px;">
        <h3 style="color: var(--secondary-bg); margin-bottom: 0.75rem;">🧪 Protocols</h3>
        <p style="font-size: 0.875rem; color: var(--neutral-muted-fg); margin: 0; line-height: 1.4;">Lab Procedures<br>Step-by-step protocols</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card" style="min-height: 120px;">
        <h3 style="color: var(--secondary-bg); margin-bottom: 0.75rem;">📊 Dataset</h3>
        <p style="font-size: 0.875rem; color: var(--neutral-muted-fg); margin: 0; line-height: 1.4;">GSE68086 Multi-Cancer<br>57,736 genes × 285 samples</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card" style="min-height: 120px;">
        <h3 style="color: var(--secondary-bg); margin-bottom: 0.75rem;">🛠️ Tools</h3>
        <p style="font-size: 0.875rem; color: var(--neutral-muted-fg); margin: 0; line-height: 1.4;">Verified Functions<br>Reliable analysis methods</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card" style="min-height: 120px;">
        <h3 style="color: var(--secondary-bg); margin-bottom: 0.75rem;">🤖 AI Engine</h3>
        <p style="font-size: 0.875rem; color: var(--neutral-muted-fg); margin: 0; line-height: 1.4;">Dynamic Code Generation<br>Novel query handling</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card" style="min-height: 120px;">
        <h3 style="color: var(--secondary-bg); margin-bottom: 0.75rem;">📈 Visualization</h3>
        <p style="font-size: 0.875rem; color: var(--neutral-muted-fg); margin: 0; line-height: 1.4;">Interactive Plotly<br>Publication-ready plots</p>
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
            # Read preview (5 rows) for display
            df_preview = pd.read_csv(st.session_state.data_path, nrows=5)
            st.dataframe(df_preview, height=200, use_container_width=True)
            
            # Get actual dataset dimensions
            df_full = pd.read_csv(st.session_state.data_path)
            total_rows = len(df_full)
            total_cols = len(df_full.columns)
            
            # Show basic stats with actual numbers
            st.markdown(f"""
            **Quick Stats:**
            - **Rows**: {total_rows:,}
            - **Columns**: {total_cols:,}
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
    
    # Show relevant examples based on user choice
    if st.session_state.user_choice == "protocol":
        st.caption("Try these protocol questions:")
        st.code("How do I perform RNA isolation?", language=None)
        st.code("Find a protocol for RNA-seq library prep and then customize it for low-input samples.", language=None)
        st.code("Generate a protocol for a western blot.", language=None)
    elif st.session_state.user_choice == "data_analysis":
        st.caption("Try these data analysis questions:")
        st.code("Create a PCA plot to visualize sample relationships", language=None)
        st.code("Generate a heatmap of the most variable genes", language=None)
        st.code("Compare gene expression between cancer types", language=None)
    else:
        st.caption("Choose your path above to see relevant examples!")

# --- Start Selection Section ---
if st.session_state.user_choice is None:
    st.markdown("## Choose Your Path")
    st.markdown("**Select what you'd like help with today:**")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="padding: 2rem 1.5rem; text-align: center;">
            <h2 style="color: var(--secondary-bg); margin-bottom: 1.5rem; font-size: 1.75rem;">🧪 Lab Protocols</h2>
            <p style="font-size: 1rem; color: var(--neutral-muted-fg); margin-bottom: 2rem; line-height: 1.5;">
                Get step-by-step laboratory protocols and procedure guidance for all your experimental needs
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("**Get Lab Protocols**", key="choose_protocol", use_container_width=True, type="primary"):
            st.session_state.user_choice = "protocol"
            st.session_state.show_questions = True
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="padding: 2rem 1.5rem; text-align: center;">
            <h2 style="color: var(--secondary-bg); margin-bottom: 1.5rem; font-size: 1.75rem;">🧬 Data Analysis</h2>
            <p style="font-size: 1rem; color: var(--neutral-muted-fg); margin-bottom: 2rem; line-height: 1.5;">
                Analyze gene expression data, create advanced visualizations, and perform comprehensive statistical analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("**Start Data Analysis**", key="choose_data", use_container_width=True, type="primary"):
            st.session_state.user_choice = "data_analysis"
            st.session_state.show_questions = True
            st.rerun()
    
    st.markdown("---")
    st.markdown("*💡 You can always switch between data analysis and protocol help in your conversation!*")

# --- Sample Questions Section (Conditional) ---
elif st.session_state.show_questions and st.session_state.user_choice:
    if st.session_state.user_choice == "data_analysis":
        # Data Analysis Questions
        if st.session_state.data_path:
            st.markdown("## Quick Start Data Analysis")
            st.markdown("**Click any question below to see BioScale in action with instant analysis:**")
            st.markdown("<br>", unsafe_allow_html=True)
            
            data_questions = [
                {
                    "title": "Dataset Overview",
                    "question": "Can you give me an overview of the dataset's structure?",
                    "description": "Get comprehensive statistics and data structure insights"
                },
                {
                    "title": "Variable Genes Heatmap", 
                    "question": "Generate a heatmap of the top 10 most variable genes across all samples.",
                    "description": "Visualize the most variable genes across all samples"
                },
                {
                    "title": "PCA Sample Relationships",
                    "question": "Create a PCA plot to visualize the relationship between the samples.",
                    "description": "Explore sample clustering and cancer type relationships"
                },
                {
                    "title": "Top Expression Analysis",
                    "question": "Find the top 5 genes with the highest average expression across all samples and list their mean values.",
                    "description": "Identify the most highly expressed genes in the dataset"
                },
                {
                    "title": "Differential Expression",
                    "question": "Create a volcano plot to show differentially expressed genes between Glioblastoma and Breast cancer samples.",
                    "description": "Compare gene expression between cancer types"
                }
            ]
            
            # Create columns for sample questions
            cols = st.columns(2)
            for i, q in enumerate(data_questions):
                with cols[i % 2]:
                    if st.button(
                        f"**{q['title']}**\n\n{q['description']}", 
                        key=f"data_q_{i}", 
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
        else:
            st.warning("**No dataset available.** Please upload a custom file or reset to the demo dataset using the sidebar to start data analysis.")
    
    elif st.session_state.user_choice == "protocol":
        # Protocol Questions
        st.markdown("## Quick Start Lab Protocols")
        st.markdown("**Click any question below to get detailed laboratory protocols:**")
        st.markdown("<br>", unsafe_allow_html=True)
        
        protocol_questions = [
            {
                "title": "RNA Extraction",
                "question": "How do I perform RNA isolation from tissue samples?",
                "description": "Get a complete RNA extraction protocol with materials and steps"
            },
            {
                "title": "RNA-seq Library Prep",
                "question": "I need a library preparation protocol for RNA-seq analysis.",
                "description": "Detailed RNA sequencing library preparation procedure"
            },
            {
                "title": "Western Blot",
                "question": "Generate a protocol for western blot analysis.",
                "description": "Complete western blot protocol from sample prep to detection"
            },
            {
                "title": "Custom Protocol",
                "question": "Create a custom protocol for protein extraction from cultured cells.",
                "description": "AI-generated custom protocol tailored to your needs"
            },
            {
                "title": "Low-Input RNA Protocol",
                "question": "Find a protocol for RNA-seq library prep and then customize it for low-input samples.",
                "description": "Specialized protocol for samples with limited RNA amounts"
            }
        ]
        
        # Create columns for protocol questions
        cols = st.columns(2)
        for i, q in enumerate(protocol_questions):
            with cols[i % 2]:
                if st.button(
                    f"**{q['title']}**\n\n{q['description']}", 
                    key=f"protocol_q_{i}", 
                    use_container_width=True,
                    help=q['question']
                ):
                    # Add the question to chat and process it
                    st.session_state.messages.append({"role": "user", "type": "text", "content": q['question']})
                    
                    # Process the question
                    with st.spinner("BioScale agent is processing..."):
                        response = run_agent(q['question'], st.session_state.data_path)
                        st.session_state.messages.append({"role": "assistant", **response})
                    
                    st.rerun()
    
    # Add option to change choice
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("**Change Selection**", key="change_choice", use_container_width=True):
            st.session_state.user_choice = None
            st.session_state.show_questions = False
            st.session_state.messages = []  # Clear chat history
            st.rerun()
    
    st.markdown("---", unsafe_allow_html=True)

# --- Main Chat Interface ---
if st.session_state.user_choice:
    if st.session_state.user_choice == "data_analysis":
        st.markdown("## Data Analysis Chat")
        st.markdown("""
        **🧬 Data Analysis Mode**
        Ask questions about your gene expression data in natural language. You can request visualizations, statistical analysis, or data exploration.
        """)
        
        # Chat input for data analysis
        if st.session_state.data_path is None:
            st.warning("**No dataset available.** Please upload a custom file or reset to the demo dataset using the sidebar.")
            prompt = st.chat_input("Please load a dataset first to start analyzing...", disabled=True)
        else:
            prompt = st.chat_input("💬 Ask about your data... (e.g., 'Create a PCA plot', 'Show me the most variable genes')")
    
    elif st.session_state.user_choice == "protocol":
        st.markdown("## Laboratory Protocol Chat")
        st.markdown("""
        **🧪 Protocol Mode**
        Ask for laboratory protocols and procedures. Get step-by-step instructions for common lab techniques or request custom protocols.
        """)
        
        # Chat input for protocols (doesn't require dataset)
        prompt = st.chat_input("💬 Ask about lab protocols... (e.g., 'How do I extract RNA?', 'Generate a western blot protocol')")
else:
    # If no choice made yet, don't show chat interface
    prompt = None

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
        st.markdown("### **Chat History**")
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
                # Enhanced plot display with robust error handling
                if "path" in message:
                    plot_path = message["path"]
                    
                    # Check for HTML plot files (Plotly) first
                    html_path = plot_path.replace('.png', '.html')
                    if os.path.exists(html_path):
                        try:
                            with open(html_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            # Ensure HTML content is valid and not empty
                            if len(html_content.strip()) > 100:  # Basic validation
                                components.html(html_content, height=600)
                                st.success("🎯 **Interactive plot displayed successfully!**")
                            else:
                                st.warning("HTML plot file appears corrupted or empty.")
                        except Exception as e:
                            st.warning(f"Could not display HTML plot: {e}")
                    
                    # Check for PNG files
                    elif os.path.exists(plot_path):
                        try:
                            file_size = os.path.getsize(plot_path)
                            if file_size > 5000:  # At least 5KB for meaningful plot
                                st.image(plot_path, use_column_width=True)
                                st.success("📊 **Plot displayed successfully!**")
                            else:
                                st.warning(f"Plot file is too small ({file_size} bytes). The analysis completed but plot may be corrupted.")
                                # Try to display anyway in case it's a valid small plot
                                st.image(plot_path, use_column_width=True)
                        except Exception as e:
                            st.error(f"Could not display plot: {e}")
                    
                    # Try absolute path variants  
                    elif os.path.exists(os.path.join("output", os.path.basename(plot_path))):
                        try:
                            abs_path = os.path.join("output", os.path.basename(plot_path))
                            st.image(abs_path, use_column_width=True)
                            st.success("📊 **Plot displayed successfully!**")
                        except Exception as e:
                            st.error(f"Could not display plot from output directory: {e}")
                    
                    else:
                        st.error(f"❌ **Plot file not found**: `{plot_path}`")
                        # Debug information
                        st.write("**Debug info:**")
                        st.write(f"- Looking for: `{plot_path}`")
                        st.write(f"- Alternative HTML: `{html_path}`")
                        st.write(f"- Current directory files:")
                        try:
                            files = os.listdir(".")
                            output_files = [f for f in files if f.endswith(('.png', '.html'))]
                            for f in output_files[:5]:  # Show first 5 matching files
                                st.write(f"  - {f}")
                        except:
                            st.write("  - Could not list directory")
                else:
                    st.warning("⚠️ **Plot was generated but no file path provided for display.**")
            elif message["type"] == "plotly":
                st.markdown(message["content"])
                try:
                    if "html" in message and message["html"]:
                        # Validate HTML content before displaying
                        html_content = message["html"]
                        if len(html_content.strip()) > 100:  # Basic validation
                            components.html(html_content, height=600)
                            st.success("🎯 **Interactive Plotly visualization displayed successfully!**")
                        else:
                            st.warning("Plotly HTML content appears empty or corrupted.")
                    else:
                        st.error("❌ **Plotly data missing or invalid.**")
                except Exception as e:
                    st.error(f"Could not display Plotly visualization: {e}")
                    # Fallback: try to show raw HTML if available
                    if "html" in message:
                        with st.expander("Debug: View Raw HTML"):
                            st.code(message["html"][:500] + "..." if len(message["html"]) > 500 else message["html"])
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
            elif message["type"] == "protocol":
                display_protocol(message["content"])
            elif message["type"] == "error":
                st.error(message["content"])
else:
    # Show helpful message when no chat history exists based on user choice
    if st.session_state.user_choice == "data_analysis":
        st.markdown("### **Start Your Data Analysis**")
        st.info("**Use the chat box above to ask questions about your gene expression data** or try one of the sample questions!")
        
        # Show data analysis examples
        st.markdown("**💡 Example Data Analysis Questions:**")
        data_examples = [
            "What are the top 10 most variable genes in this dataset?",
            "Create a volcano plot comparing two cancer types",
            "Generate a PCA plot showing sample relationships",
            "Show me a heatmap of the correlation between samples",
            "Perform differential expression analysis between conditions",
            "What are the basic statistics of this dataset?"
        ]
        for example in data_examples:
            st.markdown(f"• *{example}*")
    
    elif st.session_state.user_choice == "protocol":
        st.markdown("### **Start Getting Lab Protocols**")
        st.info("**Use the chat box above to ask for laboratory protocols and procedures** or try one of the sample questions!")
        
        # Show protocol examples
        st.markdown("**💡 Example Protocol Questions:**")
        protocol_examples = [
            "How do I perform RNA isolation from tissue samples?",
            "Generate a protocol for western blot analysis",
            "I need a library preparation protocol for RNA-seq",
            "Create a custom protocol for protein extraction",
            "Show me a protocol for cell culture maintenance",
            "How do I prepare samples for mass spectrometry?"
        ]
        for example in protocol_examples:
            st.markdown(f"• *{example}*")
    
    else:
        st.markdown("### **Welcome to BioScale**")
        st.info("**Choose your path above to get started** - select either Data Analysis or Lab Protocols to see relevant examples and begin!")