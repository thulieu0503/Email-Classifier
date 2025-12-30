import streamlit as st
import joblib

# Load models
models = {
    "Naive Bayes (BoW)": joblib.load("model/naive_bayes_bow_model.pkl"),
    "Logistic Regression (BoW)": joblib.load("model/logistic_regression_bow_model.pkl"),
    "Naive Bayes (TF-IDF)": joblib.load("model/naive_bayes_tfidf_model.pkl"),
    "Logistic Regression (TF-IDF)": joblib.load("model/logistic_regression_tfidf_model.pkl"),
}

# Load vectorizers
vectorizers = {
    "BoW": joblib.load("vectorizer/bow.pkl"),
    "TF-IDF": joblib.load("vectorizer/tfidf.pkl"),
}

label_map = {
    0: "Ham (Email thường)",
    1: "Spam 🚫"
}

st.title("📧 Email Spam Classification")

email_text = st.text_area("Nhập nội dung email:")

model_name = st.selectbox("Chọn mô hình:", list(models.keys()))

if st.button("Dự đoán"):
    if email_text.strip() == "":
        st.warning("⚠️ Vui lòng nhập nội dung email")
    else:
        model = models[model_name]

        # Chọn vectorizer theo tên model
        if "BoW" in model_name:
            vectorizer = vectorizers["BoW"]
        else:
            vectorizer = vectorizers["TF-IDF"]

        # Chuyển email sang vector
        X = vectorizer.transform([email_text])

        # Lấy xác suất dự đoán
        proba = model.predict_proba(X)[0]  # trả về mảng [xác suất ham, xác suất spam]
        labels = model.classes_  # ['ham', 'spam']

        # Lấy xác suất spam
        spam_proba = proba[list(labels).index('spam')]

        # So sánh với ngưỡng để dự đoán
        threshold = 0.5
        if spam_proba >= threshold:
            st.error(f"📌 Kết quả: **Spam 🚫** ({spam_proba*100:.2f}%)")
        else:
            st.success(f"📌 Kết quả: **Ham (Email thường)** ({(1-spam_proba)*100:.2f}%)")

                




