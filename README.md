# Crime Rate Prediction & Analysis using K-Means Clustering

This Django web application analyzes crime datasets (such as NCRB data), groups states into crime zones using K-Means clustering, and predicts future crime trends using Polynomial Regression. It includes dataset upload, interactive analysis, map visualizations, and a modern UI.

---

## Features

- Secure login (authenticated dashboard)
- Upload structured CSV datasets
- Data cleaning and validation
- Interactive dataset preview (DataTables)
- Crime clustering using K-Means
- Risk categorization (Low, Medium, High)
- Zoomable India crime map (Folium)
- Future crime trend prediction (2025–2026)
- Auto-generated graphs stored in `/static/graphs`
- Modern responsive UI (dark theme)

---

## Technology Stack

- **Backend:** Django (Python)  
- **Frontend:** HTML, CSS, Bootstrap 5  
- **Data Science:** Pandas, NumPy, Scikit-learn, Matplotlib, Folium  
- **Version Control:** Git + GitHub  

---

## Folder Structure

```
project/
├── manage.py
├── crime_project/          # Django settings
├── core/                   # Views, URLs, templates, static, logic
├── data/                   # Uploaded datasets
├── static/                 # CSS, JS, images, generated graphs
└── templates/
    ├── login.html
    ├── index.html
    ├── upload.html
    ├── cluster.html
    └── future.html
```

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/sravanakkaladevi/CRIME-RATE-PREDICTION-ANALYSIS-USING-K-MEANS-CLUSTERING-ALGORITHM-mini-project.git
cd CRIME-RATE-PREDICTION-ANALYSIS-USING-K-MEANS-CLUSTERING-ALGORITHM-mini-project

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

# Install dependencies
pip install -r requirements.txt

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run server
python manage.py runserver
```

Visit: **http://127.0.0.1:8000/**

---

## Usage

1. Log in with your superuser credentials  
2. Upload your CSV dataset  
3. View dataset preview  
4. Analyze crime charts and patterns  
5. Run K-Means clustering and view crime zones  
6. Predict future crime trends for any state  
7. Explore interactive map zoom (Folium)

---

## Academic Notes

This project is built for academic use in Machine Learning and Data Analysis.  
Key focus: **Unsupervised Learning (K-Means)** and **Polynomial Regression**.

---

## Future Enhancements

- Add LSTM / Prophet time-series models  
- Add user registration with roles  
- Improve map visuals with Plotly  
- Deploy on Heroku or AWS  

---

## License

MIT License — free to use, modify and distribute.
