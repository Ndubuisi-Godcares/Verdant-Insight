# 🌿 Verdant Insight

**Verdant Insight** is an intelligent plant recognition application that helps users identify houseplants in real-time using image uploads, snapshots, or live camera feeds. Built with deep learning and powered by TensorFlow/Keras and Streamlit, this app enables instant plant classification using a robust EfficientNet-based model trained on over 14,000 labeled images.

[🌿 Launch Verdant Insight](https://verdant-insight-v84tnlommdoot4agzycacm.streamlit.app)

![Verdant Insight Demo](https://github.com/user-attachments/assets/c47bf64c-934b-4601-97d3-5aa64b8f1e48)

---

## 💡 Why Verdant Insight?

Many people—especially beginners—struggle to identify or care for their houseplants. Whether it’s for better plant care, curiosity, or quick identification, Verdant Insight simplifies the process through real-time, AI-powered recognition. With just a photo, users can instantly learn about their plant species, helping them become more informed and confident plant owners.

Beyond personal utility, **Verdant Insight has broader significance in green technology and AI research**. It serves as a foundation for:

- 🌾 **Smart agriculture** tools for automated plant monitoring  
- 📱 **Mobile applications** for plant disease detection  
- 📚 **Educational platforms** for botanical learning  
- 🌍 **Environmental systems** for biodiversity tracking

This project demonstrates the real-world potential of **machine learning and computer vision in sustainability-focused solutions**.

---

## 🚀 Features

- 🌱 **Houseplant Recognition**  
  Identify various houseplants with high accuracy using transfer learning.

- 📸 **Flexible Image Input**  
  Upload images, take snapshots, or use live camera capture for real-time identification.

- 🌐 **Deployed with Streamlit Cloud**  
  Access the app from anywhere without local setup.

- 🧠 **EfficientNet Backbone**  
  Fine-tuned for superior performance on image classification tasks.

---

## 🧪 Tech Stack

- **Framework:** Streamlit  
- **Model:** TensorFlow/Keras (EfficientNetB0)  
- **Language:** Python  
- **Deployment:** Streamlit Cloud  
- **Version Control:** Git & GitHub

---

## 📁 Dataset

- **Size:** 14,000+ images  
- **Type:** Labeled houseplant images  
- **Source:** Custom curated dataset (not publicly available)

---

## 🧠 Model Info

- **Architecture:** EfficientNetB0  
- **Size:** ~133MB  
- **Note:** Due to its size, the model may take a few seconds to load when the app starts. Streamlit automatically handles lazy loading after deployment.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/verdant-insight.git
cd verdant-insight
pip install -r requirements.txt
