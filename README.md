# 🧭 Career Path Recommender

An interactive **Streamlit** application that recommends the best career path based on your skills. The app uses a **deep learning model (PyTorch MLP)** trained on skill-to-career mappings to provide intelligent career recommendations.

## 🚀 Features

- **Yes/No Quiz Interface**: Answer 27 simple plain-language questions about your interests
- **AI-Powered Recommendations**: Uses a trained PyTorch MLP model with TF-IDF vectorization
- **Top 3 Career Matches**: Displays the most suitable careers with confidence scores
- **6 Career Categories**:
  - Artificial Intelligence
  - Data Science
  - Development
  - Security
  - Software Development and Engineering
  - User Experience (UX) and UI Design

## 📋 Prerequisites

- Python 3.9+
- pip (Python package manager)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/career-path-recommender.git
   cd career-path-recommender
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running Locally

```bash
streamlit run app_streamlit.py
```

The app will open in your default browser at `http://localhost:8501`.

## ☁️ Deploying to Streamlit Cloud

### Step 1: Push to GitHub

1. Create a new repository on [GitHub](https://github.com)
2. Initialize git and push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/career-path-recommender.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository, branch (`main`), and set the main file path to `app_streamlit.py`
5. Click **"Deploy"**

### Step 3: Configure (if needed)

- The app uses **relative paths** for all model artifacts — no additional configuration needed
- All dependencies are listed in `requirements.txt`
- The model files (~22MB) are included in the repository

## 📁 Project Structure

```
├── app_streamlit.py              # Main Streamlit application (entry point)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── artifacts/
│   ├── best_model.pt             # Trained PyTorch model weights (~22MB)
│   ├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
│   ├── label_encoder.pkl         # Label encoder for career categories
│   └── model_meta.json           # Model metadata (input dim, classes, etc.)
├── Dataset/
│   └── Career_Dataset.xlsx       # Training dataset (not needed for deployment)
├── outputs/                      # Training visualizations (not needed for deployment)
└── career_recommender.ipynb      # Training notebook (not needed for deployment)
```

## 🧠 Model Architecture

The model is a **Multi-Layer Perceptron (MLP)** with:
- Input layer: TF-IDF features (up to 5000)
- Hidden layers: 1024 → 512 → 256 → 128 neurons
- Output layer: 6 career classes
- Regularization: BatchNorm, Dropout (0.4/0.3/0.2), Weight Decay
- Best validation accuracy: **~96.2%**

## 🔒 Security

- No hardcoded API keys or secrets
- No sensitive data in the repository
- All paths are relative

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
