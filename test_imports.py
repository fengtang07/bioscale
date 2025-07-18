#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed
Run this before deploying to catch import errors early
"""

def test_imports():
    """Test all required imports for BioScale app"""
    print("Testing imports...")
    
    try:
        # Core libraries
        import pandas as pd
        print("✅ pandas")
        
        import numpy as np
        print("✅ numpy")
        
        import scipy
        print("✅ scipy")
        
        # Visualization libraries
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
        
        import seaborn as sns
        print("✅ seaborn")
        
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ plotly")
        
        # ML libraries
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        print("✅ scikit-learn")
        
        # LangChain and OpenAI
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        print("✅ langchain")
        
        from pydantic import BaseModel
        print("✅ pydantic")
        
        # Streamlit
        import streamlit as st
        print("✅ streamlit")
        
        # Built-in libraries
        import zipfile
        import os
        print("✅ built-in libraries")
        
        print("\n🎉 All imports successful! App should work correctly.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1) 