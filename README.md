🚔 Crime Rate Prediction & Analysis using K-Means Clustering Algorithm
🧾 Project Overview

This Django-based web application lets users upload crime datasets (like those from the NCRB), perform exploratory data analysis, visualize results, and group cities or states using K-Means Clustering.
It also includes a trend prediction module that estimates future crime patterns using Polynomial Regression.
Goal: Identify high, medium, and low crime zones and forecast possible future trends.

🎯 Features

Secure user login (only authenticated users can access analytics)

Upload any structured CSV dataset

Data cleaning and preprocessing (handles missing or malformed entries)

Visualization dashboards:

📊 Crime counts by city/state

🧩 K-Means clustering by crime, weapon, and gender

🔮 Future crime trend prediction (Polynomial Regression)

Modern responsive UI with dark theme and interactive graph previews

🛠 Technology Stack

Backend: Django (Python)

Frontend: HTML, CSS, Bootstrap 5

Data Science: Pandas, NumPy, Scikit-learn, Matplotlib

Version Control: Git + GitHub

Dataset Source: NCRB / Public crime datasets

📁 Repository Structure
crime_project/
├── manage.py
├── crime_project/          # Django settings
├── core/                   # Main app (views, urls, templates, static)
├── data/                   # Uploaded CSV datasets
├── static/                 # CSS, JS, images, graphs
└── templates/
    ├── login.html
    ├── index.html
    ├── upload.html
    ├── analysis.html
    ├── cluster.html
    └── future.html

🚀 Setup & Run Locally
# Clone repository
git clone https://github.com/sravanakkaladevi/CRIME-RATE-PREDICTION-ANALYSIS-USING-K-MEANS-CLUSTERING-ALGORITHM.git
cd CRIME-RATE-PREDICTION-ANALYSIS-USING-K-MEANS-CLUSTERING-ALGORITHM

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate        # On Windows
# source venv/bin/activate   # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Apply migrations & collect static files
python manage.py migrate
python manage.py collectstatic

# Create a superuser for login
python manage.py createsuperuser

# Run server
python manage.py runserver


Then visit: http://127.0.0.1:8000/

📄 Usage

Log in using your superuser credentials

Upload the dataset via Upload tab

View graphs in Analysis

Check crime grouping via Cluster (K-Means)

Predict next-year trends in Future

Log out when done

🎓 Academic Context

This project was developed as part of an academic course on Machine Learning & Data Analysis.
Main focus: Unsupervised Learning (K-Means Clustering) for pattern identification.
The Polynomial Regression module provides supplementary predictive insight.

🧬 Future Enhancements

Integrate deep-learning time series models (Prophet/LSTM)

Add user registration & role-based access

Include real-time map visualizations (Leaflet / Plotly)

Deploy using Heroku or AWS

📜 License

Distributed under the MIT License — free to use, modify, and share with attribution.
