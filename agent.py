import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

# Import the verified tools defined in tools.py
from tools import summarize_expression_data, plot_gene_expression_heatmap

# --- 1. Define and Map the Agent's Tools ---
verified_tools = [summarize_expression_data, plot_gene_expression_heatmap]
tool_map = {tool.name: tool for tool in verified_tools}


# --- 2. Define the Smart Router Logic ---
# The router decides whether to use a verified tool or the generative AI path.

class RouteQuery(BaseModel):
    """Routes the user's query to the appropriate tool or to the AI code generator."""
    destination: Literal[
        "summarize_expression_data", "plot_gene_expression_heatmap", "generative_python_coder"] = Field(
        description="The destination to route the query to. 'generative_python_coder' is the fallback for novel or complex queries."
    )


# Use a capable model for routing decisions (e.g., gpt-4o)
router_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create a structured output chain that forces the LLM to output in the RouteQuery format
structured_llm_router = router_llm.with_structured_output(RouteQuery)

# Create the prompt template for the router.
# The descriptions of the tools are crucial for the LLM to make the right choice.
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a smart router for a biomedical data analysis agent. Your job is to route user queries to the appropriate tool or to the generative AI coder.

Available tools:
- summarize_expression_data: Use for requests to summarize, describe, or get basic statistics about gene expression data
- plot_gene_expression_heatmap: Use for requests to create heatmaps of specific genes (requires gene IDs like ENSG...)
- generative_python_coder: Use for any other complex analysis, custom plots, or novel queries

Route to 'generative_python_coder' when:
- The query is complex or novel
- The user wants custom analysis not covered by verified tools
- The query requires data manipulation or custom visualizations
"""),
    ("human", "Query: {query}")
])

# The complete router chain
router_chain = router_prompt | structured_llm_router

# --- 3. Define the Generative AI Coder Path ---

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
- PREFER Plotly (px/go) for interactive plots - better user experience
- For volcano plots: use go.Scatter with hover data showing gene IDs
- For PCA plots: use px.scatter with detailed sample classification and variance explained
- For static plots: use matplotlib (plt) and save to 'output/generated_plot.png'
- Plotly plots: save as HTML to 'output/generated_plot.html'

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
3. For interactive plots, save as HTML: fig.write_html('output/generated_plot.html')
4. For static plots, save as PNG: plt.savefig('output/generated_plot.png')
5. Create the 'output' directory if it doesn't exist
6. Store final results in a variable called 'result_summary' for display
7. Access gene data using df.loc['GENE_ID'] not df['GENE_ID']
8. For volcano plots: use raw p-values (scipy.stats.ttest_ind), optimize with vectorization
9. Filter out genes with zero variance and low expression to avoid errors
10. ALWAYS identify actual sample names from df.columns, never use placeholder names
11. For large analyses, use subset of genes (df.iloc[:5000]) and mention this in results
12. For PCA plots: use the provided classify_sample_detailed() function, include variance explained, and use distinct colors/symbols
13. Always add hover information showing sample names and detailed group classifications  
14. For PCA: use px.scatter with symbol parameter for additional visual distinction between groups
15. Use discrete color sequences like px.colors.qualitative.Set3 for better color separation

Example code structure:
```python
import os
os.makedirs('output', exist_ok=True)

# Your analysis code here
# ...

# If creating a plot:
plt.savefig('output/generated_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```"""),
    ("human", "Query: {query}\nDataset path: {file_path}")
])

coder_llm = ChatOpenAI(model="gpt-4o", temperature=0)
generative_coder_chain = generative_coder_prompt | coder_llm


# --- 4. Define the Main Agent Logic ---

def run_agent(query: str, file_path: str) -> Dict[str, Any]:
    """
    The main entry point for the BioScale agent.
    It routes the query and executes the appropriate path.
    """
    # Step 1: Route the query
    routing_decision = router_chain.invoke({"query": query})
    destination = routing_decision.destination
    print(f"AGENT: Routing decision: '{destination}'")

    # Step 2: Execute the chosen path
    if destination in tool_map:
        # --- Verified Tool Path ---
        tool_to_use = tool_map[destination]

        # For plotting, we need to extract gene IDs from the query
        if destination == "plot_gene_expression_heatmap":
            # Simple regex to find gene IDs (e.g., ENSG...)
            gene_ids = re.findall(r'ENSG\d+', query)
            if not gene_ids:
                return {"type": "error",
                        "content": "Could not find any gene IDs (e.g., 'ENSG...') in your query for plotting."}
            result = tool_to_use.invoke({"file_path": file_path, "gene_ids": gene_ids})
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
            # Clean up any existing plots to avoid showing stale images
            for plot_file in ["output/generated_plot.png", "output/generated_plot.html"]:
                if os.path.exists(plot_file):
                    os.remove(plot_file)
                
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
                    return {
                        "type": "plot",
                        "content": str(result_data),
                        "path": png_plot_path
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
                elif os.path.exists(png_plot_path):
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