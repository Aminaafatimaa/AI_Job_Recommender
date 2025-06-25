# 🧠 AI Job Recommender

An **AI-powered career recommendation system** that uses a quiz to assess a user's interests, aptitude, and personality — and then recommends the most suitable job role.

## 🚀 Live Demo
🌐 [View the Live App on Render](https://ai-job-recommender-weyb.onrender.com)

## 📌 Features
- ✨ Quiz-based career guidance
- 🧠 Machine Learning-powered predictions
- 📊 Dataset-based model trained on 100+ responses
- 🔥 FastAPI backend
- ☁️ Deployed on Render

## 📁 Project Structure

📂 AI_Job_Recommender
├── main.py # FastAPI backend with endpoints
├── train_ai_model.ipynb # Jupyter notebook for model training
├── model.pkl # Saved ML model
├── label_encoders.pkl # Encoders for categorical data
├── job_quiz_dataset_100_final.csv
├── requirements.txt # Python dependencies
└── README.md

## ⚙️ How It Works

1. User takes a short quiz.
2. Answers are encoded and passed to the ML model.
3. The model predicts the best-fit job role.
4. Result is shown to the user instantly!

## 💻 Tech Stack

| Tech        | Used For                  |
|-------------|---------------------------|
| Python      | Core logic and ML         |
| FastAPI     | API creation              |
| Pandas, Sklearn | Data processing & ML   |
| Render      | Cloud deployment          |
| GitHub      | Version control           |

## 🔧 Setup Instructions

```bash
git clone https://github.com/Aminaafatimaa/AI_Job_Recommender.git
cd AI_Job_Recommender
pip install -r requirements.txt
python main.py

## 👩‍💻 Author

- Amina Fatima – [GitHub Profile](https://github.com/Aminaafatimaa)


