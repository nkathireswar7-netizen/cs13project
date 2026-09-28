import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Adaptive AI Network Security Analyst",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ Adaptive AI Network Security Analyst")
st.write("AI-powered Network Intrusion Detection Dashboard")

st.divider()

# Dataset upload section placed first to allow metric updates
st.subheader("📂 Upload Network Flow Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

# Process dataset if uploaded, otherwise use default empty stats
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    total_flows = len(df)
    
    # Check for "Malicious" or "ATTACK" prediction variants
    threats_df = df[df["Prediction"].astype(str).str.upper().isin(["MALICIOUS", "ATTACK"])]
    threats_count = len(threats_df)
    
    # Check for "High" or "HIGH" risk levels
    high_risk_count = len(df[df["Risk"].astype(str).str.upper() == "HIGH"])
else:
    df = None
    total_flows = 0
    threats_count = 0
    high_risk_count = 0
    threats_df = pd.DataFrame()

# Dashboard metrics (now dynamically updated)
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Flows", total_flows)
col2.metric("Threats", threats_count)
col3.metric("High-Risk Incidents", high_risk_count)
col4.metric("False Positives", "0")

st.divider()

if df is not None:
    st.success("Dataset uploaded successfully!")

    st.subheader("📊 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
else:
    st.info("Upload a network-flow CSV dataset to begin.")

st.divider()

# Suspicious flows (uses uploaded threats if available, otherwise sample fallback)
st.subheader("🚨 Suspicious Network Flows")

if df is not None and not threats_df.empty:
    st.dataframe(
        threats_df,
        use_container_width=True
    )
elif df is not None and threats_df.empty:
    st.info("No suspicious network flows detected in the uploaded dataset.")
else:
    # Static sample data display prior to uploading a file
    sample_data = pd.DataFrame({
        "Time": ["10:31", "10:32", "10:35"],
        "Source": ["192.168.1.10", "192.168.1.20", "192.168.1.15"],
        "Destination": ["10.0.0.5", "10.0.0.8", "10.0.0.10"],
        "Prediction": ["ATTACK", "BENIGN", "ATTACK"],
        "Risk": ["HIGH", "LOW", "HIGH"]
    })
    st.dataframe(
        sample_data,
        use_container_width=True
    )

st.divider()

# AI Investigation
st.subheader("🔍 AI Investigation")

if st.button("Investigate Selected Alert"):

    st.warning("⚠️ ATTACK DETECTED")

    st.write("### ML Confidence")
    st.write("96%")

    st.write("### Observed Evidence")

    st.write("- Suspicious network flow")
    st.write("- Abnormal packet activity")
    st.write("- Unusual communication pattern")

    st.write("### AI Interpretation")

    st.write(
        "The observed network behaviour is consistent "
        "with potentially malicious activity."
    )

    st.write("### Risk")

    st.error("HIGH")

    st.write("### Recommended Investigation")

    st.write("1. Examine the source host")
    st.write("2. Check related network connections")
    st.write("3. Review security logs")

st.divider()

# Analyst feedback
st.subheader("👨‍💻 Analyst Feedback")

col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Confirm Incident"):
        st.success("Incident confirmed.")

with col2:
    if st.button("❌ False Positive"):
        st.info("Marked as false positive.")

st.divider()

# Incident history
st.subheader("📋 Incident History")

history = pd.DataFrame({
    "Incident": ["#001", "#002", "#003"],
    "Attack Type": ["Reconnaissance", "DoS", "Unknown"],
    "Risk": ["HIGH", "HIGH", "MEDIUM"],
    "Analyst Decision": [
        "Confirmed",
        "Confirmed",
        "False Positive"
    ]
})

st.dataframe(
    history,
    use_container_width=True
)