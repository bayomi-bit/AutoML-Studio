import streamlit as st
import pandas as pd

from core.orchestrator import AutoMLPipeline

st.set_page_config(page_title="AutoML Studio", layout="wide")

st.title("🤖 AutoML Studio")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button(" Run AutoML"):

        pipeline = AutoMLPipeline()
        output = pipeline.run(df)

        st.subheader(" Results")
        st.write(output["results"])

        st.success(f" Best Model: {output['best_model']}")