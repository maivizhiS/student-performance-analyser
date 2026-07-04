# 🎓 Student Academic Performance Analyser

🔗 **Live Demo:** https://student-performance-analyser-472026.streamlit.app/

An interactive Streamlit dashboard for exploring and predicting student academic performance based on demographic and preparation factors.

## Features
- 📊 Exploratory data analysis across gender, parental education, exam preparation, and family income
- 📈 Advanced visualizations: correlation heatmaps, box plots, scatter plots, pair plots
- 🤖 Machine Learning prediction (Logistic Regression & Random Forest) to predict Pass/Fail outcomes
- 📋 Model performance evaluation with confusion matrix and feature importance
- 💡 Key insights and recommendations based on the data

## Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## How to Run

1. Clone this repository
```bash
git clone https://github.com/maivizhiS/student-performance-analyser.git
cd student-performance-analyser
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

## Dataset
The dataset (`StudentsPerformance.csv`) contains student scores in Math, Reading, and Writing along with demographic details such as gender, parental education level, family income level, and exam preparation status.
