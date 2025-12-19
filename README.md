
# 🎙️ AI Enabled Real-Time AI Sales Call Assistant for Enhanced Conversation Strategies

An **offline AI-powered system** that analyzes sales calls in real time to extract customer sentiment, intent, and key entities, and provides intelligent conversation strategies using a local large language model 

---

## 📌 Overview

Sales calls are critical for business success, but understanding customer intent and responding effectively in real time is challenging. This project presents an **AI-enabled real-time sales call assistant**.

The system processes live or recorded customer calls, identifies speakers, performs natural language processing, and generates actionable sales guidance such as:
- Next best questions
- Objection handling responses
- Product recommendations

---

## 🚀 Key Features

- 🎙️ Live audio recording and audio file upload  
- 📝 Automatic speech-to-text transcription 
- 👥 Speaker diarization (Customer vs Agent identification)  
- 😊 Sentiment analysis (Positive / Neutral / Negative)  
- 🎯 Intent detection (Discount request, inquiry, complaint, etc.)  
- 🧩 Named Entity Recognition (Product, Brand, Price, etc.)  
- 🤖 AI-driven sales suggestions 
- 🔒 Fully offline and privacy-preserving  
- 🖥️ Interactive web interface built with **Streamlit**

---

## 🧰 Technology Stack

| Component | Technology |
|--------|-----------|
| Frontend | Streamlit |
| Speech-to-Text | Whisper |
| Speaker Identification | Resemblyzer |
| Sentiment Analysis | HuggingFace |
| Intent Detection | BART Zero-Shot |
| Entity Extraction | spaCy |
| AI Reasoning | PHI-3 (via Ollama) |
| Deployment | Fully Local |

---

## 🧪 Example Output

**Customer Sentence:**  
> *“Do you have any discount on the Samsung soundbar?”*

**Analysis:**
- Sentiment: Neutral  
- Intent: Discount request  
- Entities: Samsung, Soundbar  

**AI Sales Suggestions:**
- Next Question: Ask about budget range  
- Objection Handling: Offer value-based bundles  
- Product Recommendation: Samsung Q-Series Soundbar  

---




