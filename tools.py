import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from langchain_core.tools import tool

# Ensure an output directory exists for saving generated plots
if not os.path.exists("output"):
    os.makedirs("output")


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