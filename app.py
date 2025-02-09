import streamlit as st
from main import (
    pdf_to_text_parallel as pdf_to_text,
    chunk_text,
    add_to_faiss_index,
    query_faiss,
    query_deepseek,
    processed_files,
)
import os

# Streamlit UI
st.title("Document Query Application")
st.sidebar.header("Upload PDFs")

# Upload PDFs
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

data_dir = "data"

# Check if the data directory is empty
is_data_dir_empty = len(os.listdir(data_dir)) == 0

# Display uploaded files in the sidebar
if not is_data_dir_empty:
    uploaded_files_list = os.listdir(data_dir)
    st.sidebar.subheader("Uploaded Files:")
    for file_name in uploaded_files_list:
        st.sidebar.write(file_name)


# File upload and processing logic
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        for uploaded_file in uploaded_files:
            # Check if the file already exists in the data directory
            file_path = os.path.join(data_dir, uploaded_file.name)
            if uploaded_file.name in processed_files:
                st.warning(f"File '{uploaded_file.name}' already exists in the data directory. Skipping processing.")
                continue

            # Save the uploaded file to the data directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the PDF
            text = pdf_to_text(file_path)
            chunks = chunk_text(text)
            add_to_faiss_index(chunks)
            processed_files.append(uploaded_file.name)

        st.success("PDFs processed and added to FAISS index.")


# If the data directory is empty, prompt the user to upload files before querying
if is_data_dir_empty:
    st.warning("The document repository is empty. Please upload PDF files to query.")


st.header("Search the Document Repository")

query = st.text_input("Enter your query:")
if query and not is_data_dir_empty:
    with st.spinner("Searching FAISS index..."):
        results, distances = query_faiss(query)

        # Check if results are valid
        if all(idx == -1 for idx in results):
            st.warning("No matches found in the FAISS index for your query.")
        else:
            # Process and display results
            st.session_state.results = results
            st.session_state.distances = distances
            st.subheader(f"Top {len(results)} results for your query:")
            for i, (result, distance) in enumerate(zip(results, distances[0])):
                st.write(f"**Result {i + 1}:** {result}")
                st.write(f"**Distance:** {distance}")
                st.write("---")

            # Query DeepSeek for refined results
            context = " ".join(results)
            with st.spinner("DeepSeek is thinking..."):
                response = query_deepseek(context, query)
                st.subheader("DeepSeek Response:")
                st.write(response)

                # Clear results and distances
                st.session_state.results = []
                st.session_state.distances = []

elif query and is_data_dir_empty:
    st.warning("Please upload and process PDFs before querying.")
