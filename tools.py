import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from langchain_core.tools import tool

# Added imports for protocol functionality
from schemas import Protocol
from protocol_db import load_protocol_library

# Ensure an output directory exists for saving generated plots
if not os.path.exists("output"):
    os.makedirs("output")

# --- Verified Protocol Tool (New Section) ---
VERIFIED_PROTOCOLS = load_protocol_library()


@tool
def summarize_expression_data(file_path: str) -> Dict[str, Any]:
    """
    Reads a CSV file of gene expression data and returns a summary.
    The summary includes the dataset's shape (number of genes/samples) and a preview of the first 5 gene IDs.
    Use this tool when the user asks for a summary, overview, or general information about the data.

    Args:
        file_path (str): The local path to the gene expression CSV file.
    """
    if not os.path.exists(file_path):
        return {"status": "Error", "message": f"File not found at path: {file_path}"}

    try:
        df = pd.read_csv(file_path, index_col=0)
        # Gene IDs are now the index
        gene_ids = df.index.tolist()
        summary = {
            "status": "Success",
            "message": "Data summary generated successfully.",
            "details": {
                "genes": df.shape[0],
                "samples": df.shape[1],
                "gene_id_preview": gene_ids[:5],
                "sample_preview": df.columns[:5].tolist()
            }
        }
        return summary
    except Exception as e:
        return {"status": "Error", "message": f"Failed to summarize data: {str(e)}"}


@tool
def plot_gene_expression_heatmap(file_path: str, gene_ids: List[str]) -> Dict[str, str]:
    """
    Generates and saves a heatmap for a specific list of gene IDs from a gene expression CSV file.
    Use this tool when the user asks to plot, visualize, or create a heatmap for one or more specific genes.
    The user must provide the gene IDs (e.g., ENSG00000134824) in the query.

    Args:
        file_path (str): The local path to the gene expression CSV file.
        gene_ids (List[str]): A list of gene IDs to include in the heatmap.
    """
    if not os.path.exists(file_path):
        return {"status": "Error", "message": f"File not found at path: {file_path}"}

    try:
        df = pd.read_csv(file_path, index_col=0)
        # Gene IDs are already the index

        # Filter the dataframe to include only the requested gene IDs that are present in the data
        data_to_plot = df.loc[df.index.isin(gene_ids)]

        if data_to_plot.empty:
            return {"status": "Error",
                    "message": f"None of the provided gene IDs were found in the dataset: {gene_ids}"}

        # Create the heatmap
        plt.figure(figsize=(12, max(4, len(data_to_plot) * 0.5)))
        sns.heatmap(data_to_plot, cmap="viridis")
        plt.title(f"Expression Heatmap for {len(data_to_plot)} Gene(s)")
        plt.xlabel("Samples")
        plt.ylabel("Gene IDs")
        plt.tight_layout()

        # Save the plot to a file in the 'output' directory
        output_path = os.path.join("output", "gene_expression_heatmap.png")
        plt.savefig(output_path)
        plt.close()  # Close the plot to free up memory

        return {
            "status": "Success",
            "message": f"Heatmap generated for {len(data_to_plot)} found gene(s).",
            "plot_path": output_path
        }
    except Exception as e:
        return {"status": "Error", "message": f"Failed to generate plot: {str(e)}"}


@tool
def plot_top_variable_genes_heatmap(file_path: str, top_n: int = 10, max_samples: int = 50) -> Dict[str, str]:
    """
    Generates and saves a heatmap for the top N most variable genes from a gene expression CSV file.
    Use this tool when the user asks for a heatmap of the most variable genes.
    The data is properly preprocessed by filtering low-expression genes and selecting representative samples.

    Args:
        file_path (str): The local path to the gene expression CSV file.
        top_n (int): Number of most variable genes to include (default: 10).
        max_samples (int): Maximum number of samples to display for readability (default: 50).
    """
    if not os.path.exists(file_path):
        return {"status": "Error", "message": f"File not found at path: {file_path}"}

    try:
        df = pd.read_csv(file_path, index_col=0)
        
        # Step 1: Pre-filter genes with low expression to reduce noise
        # Only keep genes with mean expression >= 1
        filtered_df = df[df.mean(axis=1) >= 1]
        
        if filtered_df.empty:
            return {"status": "Error", "message": "No genes found with sufficient expression levels"}
        
        # Step 2: Calculate variance for each gene
        gene_variances = filtered_df.var(axis=1)
        
        # Step 3: Get the top N most variable genes
        top_variable_genes = gene_variances.nlargest(top_n).index
        top_genes_data = filtered_df.loc[top_variable_genes]
        
        # Step 4: If there are too many samples, select a representative subset
        if len(top_genes_data.columns) > max_samples:
            # Select samples with highest variance across the selected genes
            sample_variances = top_genes_data.var(axis=0)
            selected_samples = sample_variances.nlargest(max_samples).index
            top_genes_data = top_genes_data[selected_samples]
        
        # Step 5: Apply log2 transformation for better visualization (add 1 to avoid log(0))
        import numpy as np
        log_data = pd.DataFrame(
            data=np.log2(top_genes_data + 1),
            index=top_genes_data.index,
            columns=top_genes_data.columns
        )
        
        # Step 6: Create the heatmap
        plt.figure(figsize=(max(12, len(log_data.columns) * 0.3), max(6, len(log_data) * 0.8)))
        
        # Use a better colormap and add proper styling
        sns.heatmap(
            log_data, 
            cmap="RdYlBu_r",  # Red-Yellow-Blue reversed colormap
            cbar_kws={'label': 'Log2(Expression + 1)'},
            xticklabels=True if len(log_data.columns) <= 30 else False,  # Hide x-labels if too many
            yticklabels=True,
            linewidths=0.1
        )
        
        plt.title(f'Heatmap of Top {len(top_genes_data)} Most Variable Genes\n({len(top_genes_data.columns)} samples)')
        plt.xlabel("Samples")
        plt.ylabel("Gene IDs")
        plt.tight_layout()

        # Save the plot to a file in the 'output' directory
        output_path = os.path.join("output", "generated_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory

        return {
            "status": "Success",
            "message": f"Heatmap generated for top {len(top_genes_data)} most variable genes across {len(top_genes_data.columns)} samples.",
            "plot_path": output_path,
            "genes_included": top_variable_genes.tolist()
        }
    except Exception as e:
        return {"status": "Error", "message": f"Failed to generate heatmap: {str(e)}"}


@tool
def find_verified_protocol(query: str) -> str:
    """
    Searches the library of verified, pre-written lab protocols.
    Use this tool first to find reliable protocols for common procedures
    like 'RNA isolation', 'library preparation', 'RNA-seq', or 'RNA extraction'.
    """
    query = query.lower()
    for keyword, protocol in VERIFIED_PROTOCOLS.items():
        if keyword in query:
            print(f"Found verified protocol: {protocol.title}")
            # Return the protocol as a JSON string to pass to other tools or the LLM
            return protocol.model_dump_json(indent=2)
    return "No verified protocol found for that query."