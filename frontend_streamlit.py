import streamlit as st
import requests

st.set_page_config(
    page_title="AI Job Post Fraud Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Job Posting Fraud Verification Engine")
st.markdown("Upload a screenshot (**PNG, JPG**) or a document (**PDF**) of a job posting to run real-time safety classification rules and transformer model inference.")

uploaded_file = st.file_uploader(
    "Choose a file", 
    type=["pdf", "png", "jpg", "jpeg", "jpe"],
    help="Maximum file size limit depends on your FastAPI engine configurations."
)

if uploaded_file is not None:
    st.info(f"📁 Loaded: {uploaded_file.name} ({uploaded_file.type})")
    
    if st.button("🚀 Analyze Job Posting", use_container_width=True):
        with st.spinner("Extracting text and executing deep inference layers..."):
            try:
                # Prepare binary data payload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Pointing to the local FastAPI app instance
                API_URL = "http://127.0.0.1:8000/predict"
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.write("---")
                    
                    # 1. Render Big Status Banner
                    is_fake = "FRAUDULENT" in data["final_prediction"].upper()
                    
                    if is_fake:
                        st.error(f"🚨 **Verdict: {data['final_prediction']}**")
                    else:
                        st.success(f"✅ **Verdict: {data['final_prediction']}**")
                        
                    # 2. Render Confidence Metrics Side by Side
                    col1, col2 = st.columns(2)
                    col1.metric(label="Confidence (Real)", value=f"{data['confidence_real']:.2f}%")
                    col2.metric(label="Confidence (Fake)", value=f"{data['confidence_fake']:.2f}%")
                    
                    # 3. Handle System Indicator Flag Alerts
                    if data["safety_override_triggered"]:
                        st.warning("⚠️ **Safety Override Triggered**: The raw model initially predicted this was Real, but background regex filters flagged explicit high-risk matching patterns (e.g., Telegram chat, suspicious payment terms), forcing a Fraud classification.")
                        
                    # 4. Show Extracted Content Preview Window
                    with st.expander("📝 View Extracted Text Snippet"):
                        st.text_area("OCR / PDF Parser Read-out:", value=data["extracted_text_snippet"], height=150, disabled=True)
                        
                else:
                    error_details = response.json().get("detail", "Unknown server processing error.")
                    st.error(f"❌ Backend Rejection Error: {error_details}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Failed: Could not connect to your FastAPI backend server at http://127.0.0.1:8000. Ensure it is currently running.")
            except Exception as e:
                st.error(f"❌ Verification Pipeline Failed: {str(e)}")
