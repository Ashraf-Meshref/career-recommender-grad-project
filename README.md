# 🧭 Career Path Recommender

An interactive **Streamlit** web application that helps users discover their ideal career path through a simple Yes/No quiz. Instead of requiring technical jargon, the app asks plain-language questions about interests and skills, then uses a trained deep learning model to recommend the top 3 career matches.

## ✨ Features

- **Yes/No Quiz Interface** — 27 easy-to-understand questions, no technical knowledge required.
- **Deep Learning Model** — A multi-layer perceptron (MLP) trained on skill-to-career mappings with **96.2% validation accuracy**.
- **Top 3 Recommendations** — Results are ranked by confidence score and displayed with progress bars.
- **Retake & Navigate** — Users can go back to previous questions or retake the quiz at any time.

## 🧠 How It Works

1. The user answers 27 Yes/No questions about their interests and skills.
2. "Yes" answers are collected and transformed using a **TF-IDF vectorizer**.
3. The vectorized input is passed through a trained **PyTorch MLP** neural network.
4. Softmax probabilities (with temperature scaling) produce a ranked list of career matches.
5. The **top 3 careers** are displayed with confidence percentages.

### Career Categories

The model classifies users into one of six career paths:

| # | Career Path |
|---|-------------|
| 0 | Artificial Intelligence |
| 1 | Data Science |
| 2 | Development |
| 3 | Security |
| 4 | Software Development and Engineering |
| 5 | User Experience (UX) and User Interface (UI) Design |

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ashraf-Meshref/career-recommender-grad-project.git
   cd career-recommender-grad-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open your browser to the URL shown in the terminal (typically `http://localhost:8501`).

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python runtime version (for deployment)
├── README.md                   # This file
└── artifacts/
    ├── best_model.pt           # Trained PyTorch model weights
    ├── label_encoder.pkl       # Label encoder for career classes
    ├── model_meta.json         # Model metadata (input dim, classes, accuracy)
    └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
```

## 🧪 Model Details

- **Architecture**: 5-layer MLP with Batch Normalization, ReLU activations, and Dropout.
- **Input**: TF-IDF vectors (4,678 features).
- **Output**: 6 career classes.
- **Validation Accuracy**: 96.2%.
- **Temperature Scaling**: A softmax temperature of 2.5 is used to produce more realistic confidence spreads.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/)
- **Data Processing**: NumPy, Joblib

## 📄 License

This project was developed as a graduation project. All rights reserved.

## 👤 Author

**Ashraf Meshref**

- GitHub: [@Ashraf-Meshref](https://github.com/Ashraf-Meshref)
