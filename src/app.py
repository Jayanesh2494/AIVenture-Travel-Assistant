import streamlit as st
import requests
from config import FASTAPI_URL  # ✅ use config only


st.title("🧭 AIVenture - Travel Assistance")
st.markdown("Ask about any travel destination - we'll find the best suggestions for you!")


# Sidebar for File Upload
with st.sidebar:
    st.subheader("📁 Upload Travel Guide")
    uploaded_file = st.file_uploader("Upload a PDF travel guide (optional)", type="pdf")

    if uploaded_file:
        if st.button("Process Guide"):
            with st.spinner("Processing the travel guide..."):
                try:
                    files = {"file": uploaded_file}
                    response = requests.post(f"{FASTAPI_URL}/upload", files=files)

                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error("Upload failed")

                except Exception as e:
                    st.error(f"Error uploading file: {str(e)}")


# Main content
st.subheader("❓ Ask Your Question")
query = st.text_input("Enter your travel question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/ask",
                    json={"query": query}
                )

                if response.status_code == 200:
                    answer = response.json()["response"]
                    st.success("Here’s your travel plan:")
                    st.write(answer)
                else:
                    st.error("Server error! Try again later.")

            except Exception as e:
                st.error(f"Could not reach server: {e}")
    else:
        st.warning("Please enter a query.")