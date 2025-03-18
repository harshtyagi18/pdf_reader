import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.title("📘 PDF Question Answering with BERT")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Uploading and extracting text..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{BACKEND_URL}/upload/", files=files)

        if response.status_code == 200 and response.json()["status"] == "success":
            st.success("PDF uploaded successfully!")
            st.text_area("Extracted Text (First 500 characters)", response.json().get("text"))
            st.session_state["pdf_uploaded"] = True
        else:
            st.error("Failed to upload PDF")
else:
    st.session_state["pdf_uploaded"] = False

question = st.text_input("Ask a question about the document")

if st.button("Get Answer"):
    if not st.session_state.get("pdf_uploaded", False):
        st.warning("Please upload a PDF first.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            response = requests.post(f"{BACKEND_URL}/answer/", data={"question": question})
            if response.status_code == 200:
                answers = response.json()["answers"]
                for i, ans in enumerate(answers):
                    st.write(f"**Answer {i+1}:** {ans['text']} (Confidence: {ans['score']:.2f})")
            else:
                st.error("Failed to generate answer")
