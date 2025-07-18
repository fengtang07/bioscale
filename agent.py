import os
import re
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

# Import the verified tools defined in tools.py
from tools import (
    summarize_expression_data,
    plot_gene_expression_heatmap,
    plot_top_variable_genes_heatmap,  # Added for variable genes heatmap
    find_verified_protocol,  # Added
)

# --- 1. Define and Map the Agent's Tools ---
verified_tools = [summarize_expression_data, plot_gene_expression_heatmap, plot_top_variable_genes_heatmap]
tool_map = {tool.name: tool for tool in verified_tools}


# --- 2. Define the Smart Router Logic ---
# The router decides whether to use a verified tool or the generative AI path.

class RouteQuery(BaseModel):
    """Routes the user's query to the appropriate tool or to the AI code generator."""
    destination: Literal[
        "summarize_expression_data", "plot_gene_expression_heatmap", "plot_top_variable_genes_heatmap", "protocol_handler", "generative_python_coder"] = Field(
        description="The destination to route the query to. 'protocol_handler' for lab protocols, 'generative_python_coder' is the fallback for novel or complex queries."
    )


# Load environment variables from .env file (local) or st.secrets (cloud)
import os
import streamlit as st

# Try to load from .env file first (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available in cloud deployment

# Get API key from environment or Streamlit secrets
def get_openai_api_key():
    # Try environment variable first (local)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Try Streamlit secrets (cloud deployment)
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, AttributeError):
        return None

# Check for OpenAI API key and provide helpful error message
api_key = get_openai_api_key()
if not api_key:
    raise ValueError("""
    ❌ OpenAI API Key Required!
    
    For LOCAL development:
    1. Create a .env file in the project root with:
       OPENAI_API_KEY=your_api_key_here
       
    2. Or set it as an environment variable:
       export OPENAI_API_KEY=your_api_key_here
    
    For STREAMLIT CLOUD deployment:
    1. Go to your app settings in Streamlit Cloud
    2. Add to Secrets section:
       OPENAI_API_KEY = "your_api_key_here"
    
    Get your API key from: https://platform.openai.com/api-keys
    """)

# Use a capable model for routing decisions (e.g., gpt-4o)
# Explicitly pass the API key to ensure it works in both local and cloud environments
router_llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)

# Create a structured output chain that forces the LLM to output in the RouteQuery format
structured_llm_router = router_llm.with_structured_output(RouteQuery)

# Create the prompt template for the router.
# The descriptions of the tools are crucial for the LLM to make the right choice.
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a smart router for a biomedical data analysis agent. Your job is to route user queries to the appropriate tool or to the generative AI coder.

Available tools:
- summarize_expression_data: Use for requests to summarize, describe, or get basic statistics about gene expression data
- plot_gene_expression_heatmap: Use for requests to create heatmaps of specific genes (requires gene IDs like ENSG...)
- plot_top_variable_genes_heatmap: Use for requests to create heatmaps of the most variable genes (finds variable genes automatically)
- protocol_handler: Use for any laboratory protocol requests (RNA extraction, library prep, western blot, etc.)
- generative_python_coder: Use for any other complex analysis, custom plots, or novel queries

Route to 'protocol_handler' when:
- The user asks for a protocol, procedure, or method
- They mention lab techniques like "RNA extraction", "library prep", "western blot", etc.
- They want step-by-step procedures

Route to 'generative_python_coder' when:
- The query is complex or novel data analysis
- The user wants custom analysis not covered by verified tools
- The query requires data manipulation or custom visualizations
"""),
    ("human", "Query: {query}")
])

# The complete router chain
router_chain = router_prompt | structured_llm_router


# --- 3. Protocol Handler ---
def handle_protocol_request(query: str) -> Dict[str, Any]:
    """Handle protocol requests using the find_verified_protocol tool."""
    try:
        # First, try to find a verified protocol
        protocol_result = find_verified_protocol.invoke({"query": query})
        
        if protocol_result != "No verified protocol found for that query.":
            # Found a verified protocol
            return {
                "type": "protocol",
                "content": protocol_result
            }
        else:
            # No verified protocol found - generate a custom one
            return generate_custom_protocol(query)
            
    except Exception as e:
        return {"type": "error", "content": f"Protocol handler error: {str(e)}"}


def generate_custom_protocol(query: str) -> Dict[str, Any]:
    """Generate a custom protocol when no verified protocol is found."""
    protocol_llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    
    protocol_prompt = f"""
    Generate a detailed laboratory protocol for: {query}
    
    Return the protocol as a valid JSON object with this exact structure:
    {{
        "title": "Protocol Title",
        "objective": "One sentence describing the goal",
        "materials": {{
            "Reagents": "List of reagents needed",
            "Equipment": "List of equipment needed",
            "Samples": "Sample requirements"
        }},
        "steps": [
            {{
                "step_number": 1,
                "description": "Detailed description of the step",
                "duration_minutes": 10
            }}
        ]
    }}
    
    Ensure the protocol is scientifically accurate, includes safety considerations, and follows standard laboratory practices.
    """
    
    try:
        response = protocol_llm.invoke([HumanMessage(content=protocol_prompt)])
        protocol_json = response.content
        
        # Clean up the response if it contains markdown
        if '```json' in protocol_json:
            protocol_json = protocol_json.split('```json')[1].split('```')[0].strip()
        elif '```' in protocol_json:
            protocol_json = protocol_json.split('```')[1].split('```')[0].strip()
            
        return {
            "type": "protocol",
            "content": protocol_json
        }
    except Exception as e:
        return {"type": "error", "content": f"Failed to generate protocol: {str(e)}"}


# --- 4. Define the Generative AI Coder Path ---

# This prompt instructs the LLM to act as a bioinformatician and write a Python script.
generative_coder_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert bioinformatician and Python programmer. You write clean, efficient Python code to analyze gene expression data.

The user has uploaded a gene expression dataset at: {file_path}
This dataset is already loaded as a pandas DataFrame called 'df'.

Available libraries: pandas (pd), seaborn (sns), matplotlib.pyplot (plt), numpy (np), scipy.stats (stats), plotly.express (px), plotly.graph_objects (go), sklearn.decomposition.PCA (PCA), sklearn.preprocessing.StandardScaler (StandardScaler), os

The DataFrame 'df' contains gene expression data where:
- ROWS are genes (index contains gene IDs like ENSG00000134824)  
- COLUMNS are samples (column names are sample IDs)
- Dataset size: 57,736 genes × 285 samples

Sample identification patterns in this dataset:
- GBM/Glioblastoma samples: contain 'GBM' in column name
- Breast cancer samples: contain 'Breast' in column name  
- Lung cancer samples: contain 'Lung' in column name
- Use list comprehensions: [col for col in df.columns if 'GBM' in col]

OPTIMIZATION GUIDELINES:
- For large datasets (>10K genes), sample first 5000-10000 genes for speed: df.iloc[:5000]
- Use vectorized operations instead of loops when possible
- Pre-filter low-expression genes (mean < 1) to reduce noise

PLOTTING PREFERENCES:
- STRONGLY PREFER Plotly (px/go) for ALL plots - better user experience and reliability
- For heatmaps: use px.imshow() or go.Heatmap() instead of seaborn/matplotlib
- For volcano plots: use go.Scatter with hover data showing gene IDs
- For PCA plots: use px.scatter with detailed sample classification and variance explained
- For box plots: use px.box() instead of matplotlib
- ONLY use matplotlib if absolutely necessary, and if so: use plt.switch_backend('Agg') before plotting
- ALL plots should be saved as HTML: fig.write_html('output/generated_plot.html')
- Plotly plots are interactive, responsive, and display reliably in Streamlit

DETAILED SAMPLE CLASSIFICATION FOR PCA:
Use this EXACT classification function for meaningful sample grouping:

def classify_sample_detailed(sample_name):
    if 'Breast' in sample_name:
        if 'Her2-ampl' in sample_name:
            return 'Breast (Her2-ampl)'
        elif 'WT' in sample_name:
            return 'Breast (WT)'
        else:
            return 'Breast (Other)'
    elif 'GBM' in sample_name:
        if 'vIII' in sample_name:
            return 'GBM (vIII)'
        elif 'WT' in sample_name:
            return 'GBM (WT)'
        else:
            return 'GBM (Other)'
    elif 'CRC' in sample_name:
        if 'KRAS' in sample_name:
            return 'CRC (KRAS)'
        elif 'WT' in sample_name:
            return 'CRC (WT)'
        else:
            return 'CRC (Other)'
    elif 'Lung' in sample_name:
        if 'EGFR' in sample_name:
            return 'Lung (EGFR)'
        else:
            return 'Lung (Other)'
    elif 'Pancr' in sample_name:
        if 'KRAS' in sample_name:
            return 'Pancr (KRAS)'
        elif 'WT' in sample_name:
            return 'Pancr (WT)'
        else:
            return 'Pancr (Other)'
    elif 'Liver' in sample_name:
        if 'KRAS' in sample_name:
            return 'Liver (KRAS)'
        elif 'WT' in sample_name:
            return 'Liver (WT)'
        else:
            return 'Liver (Other)'
    elif 'HD' in sample_name or 'Control' in sample_name:
        return 'Healthy Controls'
    elif 'Chol' in sample_name:
        return 'Cholangiocarcinoma'
    else:
        return 'Other Samples'

Instructions:
1. Write Python code to answer the user's query
2. Use descriptive variable names and add comments
3. ALWAYS use Plotly for plots and save as HTML with EXACT filename: fig.write_html('output/generated_plot.html')
4. If matplotlib is absolutely necessary: use plt.switch_backend('Agg') first, then plt.savefig('output/generated_plot.png', dpi=300, bbox_inches='tight') and plt.close()
5. Create the 'output' directory if it doesn't exist
6. Store final results in a variable called 'result_summary' for display
7. CRITICAL: Always use the exact filename 'output/generated_plot.html' or 'output/generated_plot.png' - no other names!
8. Access gene data using df.loc['GENE_ID'] not df['GENE_ID']
9. For volcano plots: use raw p-values (scipy.stats.ttest_ind), optimize with vectorization
10. Filter out genes with zero variance and low expression to avoid errors
11. ALWAYS identify actual sample names from df.columns, never use placeholder names
12. For large analyses, use subset of genes (df.iloc[:5000]) and mention this in results
13. For PCA plots: use the provided classify_sample_detailed() function, include variance explained, and use distinct colors/symbols
14. Always add hover information showing sample names and detailed group classifications  
15. For PCA: use px.scatter with symbol parameter for additional visual distinction between groups
16. Use discrete color sequences like px.colors.qualitative.Set3 for better color separation

Example code structure:
```python
import os
import plotly.express as px
os.makedirs('output', exist_ok=True)

# Your analysis code here
# ...

# For Plotly plots (preferred) - MUST use exact filename:
fig = px.scatter(data, x='x_col', y='y_col', title='My Plot')
fig.write_html('output/generated_plot.html')  # EXACT filename required!

# Only if matplotlib is absolutely necessary:
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Use non-interactive backend
plt.figure()
# ... your matplotlib code ...
plt.savefig('output/generated_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```
    """),
    ("human", "Query: {query}\nDataset path: {file_path}")
])

coder_llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
generative_coder_chain = generative_coder_prompt | coder_llm


# --- 5. Define the Main Agent Logic ---

def run_agent(query: str, file_path: str) -> Dict[str, Any]:
    """
    The main entry point for the BioScale agent.
    It routes the query and executes the appropriate path.
    """
    try:
        # Clean up any existing plots to avoid showing stale images
        for plot_file in ["output/generated_plot.png", "output/generated_plot.html"]:
            if os.path.exists(plot_file):
                os.remove(plot_file)

        # Step 1: Route the query
        routing_decision = router_chain.invoke({"query": query})
        destination = routing_decision.destination
        print(f"AGENT: Routing decision: '{destination}'")

        # Step 2: Execute the chosen path
        if destination == "protocol_handler":
            # --- Protocol Handler Path ---
            return handle_protocol_request(query)
            
        elif destination in tool_map:
            # --- Verified Tool Path ---
            tool_to_use = tool_map[destination]

            # For plotting, we need to handle different tool requirements
            if destination == "plot_gene_expression_heatmap":
                # Simple regex to find gene IDs (e.g., ENSG...)
                gene_ids = re.findall(r'ENSG\d+', query)
                if not gene_ids:
                    return {"type": "error",
                            "content": "Could not find any gene IDs (e.g., 'ENSG...') in your query for plotting."}
                result = tool_to_use.invoke({"file_path": file_path, "gene_ids": gene_ids})
            elif destination == "plot_top_variable_genes_heatmap":
                # Extract number of genes if specified in query, default to 10
                numbers = re.findall(r'\b(\d+)\b', query)
                top_n = int(numbers[0]) if numbers else 10
                result = tool_to_use.invoke({"file_path": file_path, "top_n": top_n})
            else:
                result = tool_to_use.invoke({"file_path": file_path})

            # Format the tool output for the UI
            if result.get("status") == "Success":
                if "plot_path" in result:
                    return {"type": "plot", "content": result["message"], "path": result["plot_path"]}
                else:
                    return {"type": "json", "content": result["message"], "data": result}
            else:
                return {"type": "error", "content": f"Tool Error: {result.get('message', 'Unknown error')}"}

        elif destination == "generative_python_coder":
            # --- Generative AI Path ---
            print("AGENT: Executing Generative Coder Path...")
            generated_code_str = generative_coder_chain.invoke({"query": query, "file_path": file_path}).content

            # Extract only the Python code from the generated response
            # Look for code blocks between ```python and ```
            code_match = re.search(r'```python\n(.*?)\n```', generated_code_str, re.DOTALL)
            if code_match:
                generated_code_str = code_match.group(1)
            else:
                # Fallback: remove markdown backticks from start/end
                if generated_code_str.startswith("```python"):
                    generated_code_str = generated_code_str[9:]
                if generated_code_str.endswith("```"):
                    generated_code_str = generated_code_str[:-3]

            print(f"AGENT: Generated Code:\n---\n{generated_code_str}\n---")

            try:
                # Prepare the execution environment
                df = pd.read_csv(file_path, index_col=0)
                print(f"AGENT: Loaded data with shape: {df.shape}")

                from scipy import stats
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                local_scope = {"df": df, "pd": pd, "sns": sns, "plt": plt, "os": os, "np": np, "stats": stats, "px": px, "go": go, "PCA": PCA, "StandardScaler": StandardScaler}

                # DANGER: Executing LLM-generated code. This is not secure.
                # In a production environment, this MUST be run in a sandboxed environment.
                exec(generated_code_str, globals(), local_scope)

                # Check if a result_summary variable was created
                if 'result_summary' in local_scope:
                    result_data = local_scope['result_summary']
                    
                    # Check for plots (HTML has priority for interactive plots)
                    html_plot_path = "output/generated_plot.html"
                    png_plot_path = "output/generated_plot.png"
                    
                    if os.path.exists(html_plot_path):
                        # Interactive Plotly plot - display in Streamlit
                        with open(html_plot_path, 'r') as f:
                            html_content = f.read()
                        return {
                            "type": "plotly",
                            "content": str(result_data),
                            "html": html_content
                        }
                    elif os.path.exists(png_plot_path):
                        # Check if PNG file is valid (not empty/corrupted)
                        if os.path.getsize(png_plot_path) > 1000:  # At least 1KB
                            return {
                                "type": "plot",
                                "content": str(result_data),
                                "path": png_plot_path
                            }
                        else:
                            # PNG file exists but is too small/corrupted
                            return {
                                "type": "text", 
                                "content": str(result_data) + "\n\nPlot image file not found. The analysis may have generated an HTML plot instead."
                            }
                    else:
                        # Check if result is structured data (dict/list) for nice formatting
                        if isinstance(result_data, (dict, list)):
                            return {
                                "type": "structured_data",
                                "content": f"Analysis completed for query: '{query}'",
                                "data": result_data
                            }
                        else:
                            return {
                                "type": "text", 
                                "content": str(result_data)
                            }
                else:
                    # Fallback: check for plots only
                    html_plot_path = "output/generated_plot.html"
                    png_plot_path = "output/generated_plot.png"
                    
                    if os.path.exists(html_plot_path):
                        with open(html_plot_path, 'r') as f:
                            html_content = f.read()
                        return {
                            "type": "plotly",
                            "content": f"Interactive plot for your query: '{query}'",
                            "html": html_content
                        }
                    elif os.path.exists(png_plot_path) and os.path.getsize(png_plot_path) > 1000:
                        return {
                            "type": "plot",
                            "content": f"AI-generated plot for your query: '{query}'",
                            "path": png_plot_path
                        }
                    else:
                        return {
                            "type": "text",
                            "content": f"AI code executed successfully for query: '{query}'. Check terminal for any printed output."
                        }

            except Exception as e:
                print(f"AGENT: Error executing generated code: {e}")
                return {"type": "error",
                        "content": f"**Error executing AI-generated code:**\n\n`{e}`\n\n**Generated Code:**\n```python\n{generated_code_str}\n```"}

        else:
            return {"type": "error", "content": f"Unknown destination: {destination}"}
            
    except Exception as e:
        return {"type": "error", "content": f"Agent error: {str(e)}"}