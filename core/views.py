from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

def index(request):
    if not request.user.is_authenticated:
        return redirect('/')
    return render(request, 'index.html')


# ---------------- LOGIN PAGE ----------------
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # go to welcome page after login
        else:
            messages.error(request, "Invalid username or password.")
    return render(request, "login.html")


# ---------------- LOGOUT ----------------
def logout_view(request):
    logout(request)
    messages.info(request, "Logged out successfully.")
    return redirect('/')

# ---------------- HOME PAGE ----------------
@login_required(login_url='/')
def index(request):
    return render(request, "index.html")


# ---------------- UPLOAD PAGE ----------------
@login_required(login_url='/')
def upload_dataset(request):
    context = {}

    if request.method == "POST" and request.FILES.get("file"):
        file = request.FILES["file"]
        try:
            df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
            os.makedirs(os.path.join(settings.BASE_DIR, "data"), exist_ok=True)
            save_path = os.path.join(settings.BASE_DIR, "data", file.name)
            df.to_csv(save_path, index=False)

            context["msg"] = f"✅ Uploaded {file.name} successfully!"
            context["rows"] = df.head(5).to_html(classes="table table-striped", index=False)
        except Exception as e:
            context["msg"] = f"❌ Error reading file: {e}"

    return render(request, "upload.html", context)


# ---------------- ANALYSIS PAGE ----------------
@login_required(login_url='/')
def analyze_data(request):
    csv_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")
    if not os.path.exists(csv_path):
        messages.warning(request, "⚠️ Please upload the dataset first.")
        return redirect("/upload/")

    df = pd.read_csv(csv_path)
    graphs_dir = os.path.join(settings.BASE_DIR, "static", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Clustering
    if "Rate of Cognizable Crimes (IPC) (2022)" in df.columns and "Chargesheeting Rate (2022)" in df.columns:
        X = df[["Rate of Cognizable Crimes (IPC) (2022)", "Chargesheeting Rate (2022)"]].fillna(0)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X)
    else:
        df["Cluster"] = 0

    # Graph 1 – Bar
    plt.figure(figsize=(10, 4))
    if "2022" in df.columns:
        plt.bar(df["State/UT"], df["2022"], color='skyblue')
        plt.xticks(rotation=90, fontsize=8)
        plt.title("Crimes by State - 2022")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "top10_trend.png"))
    plt.close()

    # Graph 2 – Scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(df["Rate of Cognizable Crimes (IPC) (2022)"], df["Chargesheeting Rate (2022)"], color='orange')
    plt.title("Rate vs Chargesheeting")
    plt.xlabel("Rate (IPC)")
    plt.ylabel("Chargesheeting Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "rate_vs_chargesheet.png"))
    plt.close()

    # Graph 3 – Cluster visualization
    plt.figure(figsize=(6, 5))
    for i in df["Cluster"].unique():
        cluster_data = df[df["Cluster"] == i]
        plt.scatter(cluster_data["Rate of Cognizable Crimes (IPC) (2022)"],
                    cluster_data["Chargesheeting Rate (2022)"], label=f"Cluster {i}")
    plt.legend()
    plt.title("Clusters of States by Crime & Chargesheeting")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "clusters.png"))
    plt.close()

    # Table
    show_cols = ["State/UT", "2020", "2021", "2022",
                 "Rate of Cognizable Crimes (IPC) (2022)", "Chargesheeting Rate (2022)", "Cluster"]
    table_html = df[show_cols].head(20).to_html(classes="table table-striped", index=False)

    context = {
        "table": table_html,
        "graphs": {
            "bar": "/static/graphs/top10_trend.png",
            "scatter": "/static/graphs/rate_vs_chargesheet.png",
            "clusters": "/static/graphs/clusters.png",
        },
    }
    return render(request, "analysis.html", context)


# ---------------- CLUSTER PREDICTION ----------------
@login_required(login_url='/')
def cluster_prediction(request):
    result = None
    if request.method == "POST":
        try:
            rate = float(request.POST.get("rate"))
            charge = float(request.POST.get("charge"))
            df = pd.read_csv(os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv"))
            X = df[["Rate of Cognizable Crimes (IPC) (2022)", "Chargesheeting Rate (2022)"]].dropna()
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            kmeans.fit(X)
            cluster = kmeans.predict([[rate, charge]])[0]
            mapping = {0: "Low Crime Zone", 1: "Medium Crime Zone", 2: "High Crime Zone"}
            result = mapping.get(cluster, f"Cluster {cluster}")
        except Exception as e:
            result = f"Error: {e}"
    return render(request, "cluster.html", {"result": result})


# ---------------- FUTURE PREDICTION ----------------
@login_required(login_url='/')
def future_prediction(request):
    result = None
    if request.method == "POST":
        try:
            rate = float(request.POST.get("rate"))
            charge = float(request.POST.get("charge"))
            df = pd.read_csv(os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv"))

            features = ["Rate of Cognizable Crimes (IPC) (2022)", "Chargesheeting Rate (2022)"]
            if all(f in df.columns for f in features):
                X = df[features].dropna()
                y = df["2022"].loc[X.index]
                model = LinearRegression()
                model.fit(X, y)
                future_val = model.predict([[rate, charge]])[0]
                result = f"Predicted future crime rate (2023–24): {future_val:.2f}"
            else:
                result = "Required columns missing in dataset."
        except Exception as e:
            result = f"Error: {e}"
    return render(request, "future.html", {"result": result})
