# SONI - Machine Learning Chatbot

## 📌 Overview
SONI is an advanced chatbot built using machine learning and deep learning techniques to understand and respond to user queries effectively. The chatbot utilizes **Natural Language Processing (NLP)** and **transformer-based embeddings** to process user input and provide meaningful responses.

This project follows a structured pipeline, including **data preprocessing, model training, hyperparameter tuning, evaluation, and deployment**. Several machine learning models were tested to determine the most effective approach for intent classification and response generation.

---

## 🔥 Features
- **Trained on a dataset with:**
  - ✅ **25 unique intent tags**
  - ✅ **926 user input patterns**
  - ✅ **926 chatbot responses**
- **Pipeline Implemented:**
  - **Text Preprocessing:** Tokenization, lemmatization, stopword removal, and vectorization.
  - **Feature Extraction:** Utilized transformer-based embeddings for rich contextual understanding.
  - **Machine Learning Models:** Experimented with multiple classifiers.
  - **Evaluation Metrics:** Used accuracy, F1-score, classification report, and confusion matrix.
  - **Hyperparameter Tuning:** Improved model performance through optimization techniques.
- **Deployment Ready:** The chatbot can be integrated into real-world applications using APIs.

---

## 🚀 Project Pipeline

### 1️⃣ Data Preprocessing
- **Text Cleaning & Normalization:**
  - Tokenization of input text.
  - Lemmatization using `WordNetLemmatizer`.
  - Stopword removal to enhance efficiency.
  - Lowercasing and punctuation removal.
- **Feature Extraction:**
  - Used **768-dimensional embeddings** from Transformer models (BERT-based representation).
  - Converted textual input into numerical vectors.

### 2️⃣ Model Selection & Training
We experimented with multiple models for intent classification:
| **Model** | **Accuracy** |
|-----------|-------------|
| Logistic Regression | 39% |
| K-Nearest Neighbors | 48% |
| Decision Tree | 32% |
| Support Vector Machine (SVM) | 46% |

- **Train-Test Split:**
  - `X_train shape`: **(740, 768)**
  - `X_test shape`: **(186, 768)**
  - `y_train shape`: **(740, 768)**
  - `y_test shape`: **(186, 768)**
- The best-performing models were selected for hyperparameter tuning.

### 3️⃣ Model Evaluation
**Performance metrics used:**
- ✅ **Accuracy**
- ✅ **F1-score**
- ✅ **Classification Report**
- ✅ **Confusion Matrix**

| Model | Macro Avg | Weighted Avg | Accuracy |
|--------|-----------|-------------|-----------|
| **Logistic Regression** | 0.03 / 0.04 / 0.03 | 0.20 / 0.39 / 0.24 | **39%** |
| **K-Nearest Neighbors** | 0.09 / 0.11 / 0.09 | 0.34 / 0.48 / 0.39 | **48%** |
| **Decision Tree** | 0.05 / 0.07 / 0.06 | 0.30 / 0.32 / 0.31 | **32%** |
| **SVM** | 0.07 / 0.10 / 0.08 | 0.29 / 0.46 / 0.36 | **46%** |

---

## ⚙️ Technologies Used
- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - `NumPy`, `Pandas` → Data manipulation
  - `NLTK`, `spaCy` → NLP processing
  - `Scikit-learn`, `TensorFlow`, `PyTorch` → Model training
  - `Matplotlib`, `Seaborn` → Data visualization

---

## 💻 How to Use the Chatbot

### 🔹 1. Clone the Repository
```bash
git clone https://github.com/yourusername/SONI-Chatbot.git
cd SONI-Chatbot
```

### 🔹 2. Install Dependencies
pip install -r requirements.txt

### 🔹 3. Run the Chatbot
python chatbot.py

### 🔹 4. Interact with SONI!
You can ask the chatbot different questions, and it will respond based on its trained dataset.

### 📈 Results & Insights
  - SVM & KNN showed the best classification accuracy (~46-48%) for intent recognition.
  - Transformer embeddings improved model performance.
  - Performance can be further optimized using deep learning approaches.

### 🔮 Future Enhancements
  - Increase Dataset Size: Improve generalization by adding more training data.
  - Use LSTM/Transformer Models: Deploy advanced deep learning models for better contextual understanding.
  - API Integration: Deploy the chatbot as a REST API for web and mobile applications.
  - Real-time Learning: Implement reinforcement learning for adaptive responses.

### 📝 License
This project is licensed under the MIT License.
