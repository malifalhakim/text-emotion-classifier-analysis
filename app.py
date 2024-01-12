# Import necessary libraries
import streamlit as st
from backend.classify import classify, analyze_text
from backend.analyze import predict_all, analyze
from backend.topic import get_topic
import pandas as pd
import time

# Set the background color and page configuration
st.set_page_config(page_title="Text Emotion Classifier and Analysis", page_icon="📊", initial_sidebar_state="expanded")

# Set the page title and subtitle
st.title("Text Emotion Classifier and Analysis")

# Tambahkan deskripsi tentang model
st.info("Model ini dirancang untuk mengidentifikasi emosi dalam teks menggunakan dataset dari platform media sosial X.")
st.info("Klasifikasi emosi yang tersedia adalah: marah, takut, bahagia, cinta, sedih.")
st.info("Anda dapat menggunakan website ini untuk mengklasifikasikan satu teks atau bahkan melakukan klasifikasi untuk beberapa teks sekaligus.")
st.info("Untuk melakukan klasifikasi pada jumlah teks yang besar, Anda dapat menyimpannya dalam format CSV.")

# Add tabs for classification and analysis
tab_titles = ['Classification', 'Analysis']
tab1, tab2 = st.tabs(tab_titles)

# Classification tab
with tab1:
    st.subheader("🔮 Classification")
    text = st.text_area("You can enter text below to classify emotions", height=200, placeholder='Lewandowski bermain buruk sekali, Xavi benar-benar marah kepadanya')
    is_not_analyze = st.checkbox("Hide result explanation", value=True)

    if st.button("Classify") and len(text) != 0:
        with st.spinner("Classifying..."):
            time.sleep(2)
            result = classify(text)
            st.write("The text is classified as : ", "<span style='color: red;'>", result, "</span>", unsafe_allow_html=True)

            if not is_not_analyze:
                viz_interpret = analyze_text(text)
                st.write(viz_interpret)

# Analysis tab
with tab2:
    st.subheader("🔎 Analysis")
    st.write("You can perform text analysis in large quantities by uploading a CSV file below.")

    st.warning("The CSV file must contain a column named 'text' or which contains the text to be analyzed and no missing values.")
    hide_code = st.checkbox("Hide the analysis results in the form of a table")

    st.warning("Make sure there are at least 100 rows of data to extract the topic")
    topic_check = st.checkbox("Extract topics from the dataset (this process takes a quite long time)")
    
    file = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Analyze") and file is not None:
        df = pd.read_csv(file)

        with st.spinner("Analyzing..."):
            df = predict_all(df)
            st.pyplot(analyze(df))
            if topic_check:
                bar_chart = get_topic(df)
                st.plotly_chart(bar_chart)

        if not hide_code:
            st.write(df)

# Custom CSS styling
st.markdown('''
<style>
.stApp [data-testid="stToolbar"]{
    display:none;
}
.st-emotion-cache-1y4p8pa {padding-top: 3rem;}
</style>
''', unsafe_allow_html=True)
