# 🌿 Verdant Insight

**Verdant Insight** is an intelligent plant recognition application that helps users identify houseplants in real-time using image uploads, snapshots, or live camera feeds. Powered by deep learning (TensorFlow/Keras) and deployed with Streamlit, the app provides instant plant classification with an EfficientNet-based model trained on 14,000+ labeled images.

[🌿 Launch Verdant Insight](https://verdant-insight-v84tnlommdoot4agzycacm.streamlit.app)

![Verdant Insight Demo](https://github.com/user-attachments/assets/c47bf64c-934b-4601-97d3-5aa64b8f1e48)

---

## 💡 Why Verdant Insight?

Many plant enthusiasts—especially beginners—find it difficult to identify and care for their houseplants. **Verdant Insight** simplifies this process by providing real-time, AI-powered plant recognition through just a photo. Users can instantly identify their plant species, enabling better care and more confident plant ownership.

But beyond personal use, **Verdant Insight** demonstrates how **AI and deep learning** can address real-world challenges in green technology. This tool serves as a foundation for potential industry applications, such as:

- 🌾 **Smart agriculture**: AI-driven plant and crop monitoring  
- 📱 **Mobile health applications**: Early plant disease detection systems  
- 📚 **Educational platforms**: Botany and biodiversity learning tools  
- 🌍 **Biodiversity tracking**: AI for ecosystem monitoring and conservation

This project exemplifies the power of **machine learning and computer vision** in creating **sustainable, scalable solutions** across various industries.

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
- **Source:** [Kaggle - House Plant Species Dataset](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species)

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
```

Run in command propmt
```
streamlit run app.py
```

Note: 
Uploaded images should be of PNG file types.
