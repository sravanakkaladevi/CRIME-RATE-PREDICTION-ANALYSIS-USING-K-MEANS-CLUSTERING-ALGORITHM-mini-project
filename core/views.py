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
    request.session.flush()  # 🧹 clear dataset session
    messages.info(request, "Logged out successfully.")
    return redirect('/')

# ---------------- HOME PAGE ----------------
@login_required(login_url='/')
def index(request):
    return render(request, "index.html")
# ---------------- UPLOAD PAGE ----------------
@login_required(login_url='/')
def upload_dataset(request):
    import pandas as pd
    import os
    from django.conf import settings

    context = {}
    data_dir = os.path.join(settings.BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "NCRB_Table_1A.1.csv")

    # ---------------- STEP 1: Handle File Upload ----------------
    if request.method == "POST" and request.FILES.get("file"):
        file = request.FILES["file"]
        try:
            df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
            df.to_csv(data_path, index=False)
            request.session["dataset_uploaded"] = True  # Save session
            context["msg"] = f"✅ Uploaded {file.name} successfully!"
        except Exception as e:
            context["msg"] = f"❌ Error reading file: {e}"

    # ---------------- STEP 2: Load Existing Dataset ----------------
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)

            # Populate dropdown lists safely (limit long lists)
        
            context["cities"] = sorted(df["City"].dropna().unique().tolist()[:100])
            context["crimes"] = sorted(df["Crime Description"].dropna().unique().tolist()[:100])
            context["genders"] = sorted(df["Victim Gender"].dropna().unique().tolist()[:50])
            context["weapons"] = sorted(df["Weapon Used"].dropna().unique().tolist()[:50])

            # Apply filters
            report = request.GET.get("report")
            city = request.GET.get("city")
            crime = request.GET.get("crime")
            gender = request.GET.get("gender")
            weapon = request.GET.get("weapon")

            filtered_df = df.copy()
            if report:
                filtered_df = filtered_df[filtered_df["Report Number"].astype(str) == str(report)]
            if city:
                filtered_df = filtered_df[filtered_df["City"] == city]
            if crime:
                filtered_df = filtered_df[filtered_df["Crime Description"] == crime]
            if gender:
                filtered_df = filtered_df[filtered_df["Victim Gender"] == gender]
            if weapon:
                filtered_df = filtered_df[filtered_df["Weapon Used"] == weapon]

            # Preview first 50 rows
            context["rows"] = filtered_df.head(50).to_html(classes="table table-striped table-sm", index=False)

        except Exception as e:
            context["msg"] = f"⚠️ Could not load dataset: {e}"

    # ---------------- STEP 3: No Dataset Yet ----------------
    elif not request.session.get("dataset_uploaded"):
        context["msg"] = "⚠️ Please upload a dataset to begin."

    return render(request, "upload.html", context)

# ---------------- ANALYSIS PAGE ----------------
@login_required(login_url='/login/')
def analyze_data(request):
    csv_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")
    if not os.path.exists(csv_path):
        messages.warning(request, "⚠️ Please upload the dataset first.")
        return redirect("/upload/")

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except Exception as e:
        messages.error(request, f"Error reading dataset: {e}")
        return redirect("/upload/")

    graphs_dir = os.path.join(settings.BASE_DIR, "static", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Take smaller sample if data too large
    sample_df = df.sample(min(len(df), 20000), random_state=42)

    # 1️⃣ Top 10 Cities by Crime Count
    if "City" in sample_df.columns:
        plt.figure(figsize=(8, 4))
        sample_df["City"].fillna("Unknown").value_counts().head(10).plot(kind="bar", color="steelblue")
        plt.title("Top 10 Cities by Reported Crimes")
        plt.xlabel("City")
        plt.ylabel("Number of Crimes")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "top10_cities.png"))
        plt.close()

    # 2️⃣ Crimes by Victim Gender
    if "Victim Gender" in sample_df.columns:
        plt.figure(figsize=(4, 4))
        sample_df["Victim Gender"].fillna("Unknown").value_counts().plot(kind="pie", autopct="%1.1f%%")
        plt.title("Crimes by Victim Gender")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "gender_pie.png"))
        plt.close()

    # 3️⃣ Top Weapons Used
    if "Weapon Used" in sample_df.columns:
        plt.figure(figsize=(6, 4))
        sample_df["Weapon Used"].fillna("Unknown").value_counts().head(10).plot(kind="barh", color="firebrick")
        plt.title("Top 10 Weapons Used in Crimes")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "weapons.png"))
        plt.close()

    # 4️⃣ Crime Occurrence Time
    if "Time of Occurrence" in sample_df.columns:
        plt.figure(figsize=(8, 4))
        sample_df["Time of Occurrence"].astype(str).fillna("Unknown").value_counts().head(20).plot(kind="bar", color="orange")
        plt.title("Crimes by Time of Day (Top 20)")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "time_distribution.png"))
        plt.close()

    # Prepare preview table
    table_html = df.head(200).to_html(classes="table table-striped", index=False)

    context = {
        "table": table_html,
        "graphs": {
            "cities": "/static/graphs/top10_cities.png",
            "gender": "/static/graphs/gender_pie.png",
            "weapons": "/static/graphs/weapons.png",
            "time": "/static/graphs/time_distribution.png",
        },
    }
    return render(request, "analysis.html", context)


# ---------------- CLUSTER PREDICTION ----------------
@login_required(login_url='/login/')
def cluster_prediction(request):
    result, cluster_plot = None, None
    csv_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")

    if not os.path.exists(csv_path):
        messages.warning(request, "⚠️ Please upload the dataset first.")
        return redirect("/upload/")

    df = pd.read_csv(csv_path, on_bad_lines="skip")

    if request.method == "POST":
        city = request.POST.get("city")
        crime_type = request.POST.get("crime")

        if "City" not in df.columns or "Crime Description" not in df.columns:
            result = "❌ Dataset missing 'City' or 'Crime Description' columns."
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import TruncatedSVD

            df["combined"] = df["City"].astype(str) + " " + df["Crime Description"].astype(str)
            grouped = df.groupby("City")["combined"].apply(lambda x: " ".join(x)).reset_index()

            vec = CountVectorizer(max_features=2000, stop_words="english")
            X = vec.fit_transform(grouped["combined"])

            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            grouped["cluster"] = labels

            label_map = {0: "Low Crime Zone", 1: "Medium Crime Zone", 2: "High Crime Zone"}

            if city not in grouped["City"].values:
                result = f"⚠️ City '{city}' not found in dataset."
            else:
                cl = int(grouped.loc[grouped["City"] == city, "cluster"].values[0])
                result = f"{city} is categorized as: {label_map.get(cl, 'Cluster')}"

                # Create 2D cluster plot
                svd = TruncatedSVD(n_components=2, random_state=42)
                coords = svd.fit_transform(X)
                plt.figure(figsize=(6, 4))
                for c in range(3):
                    plt.scatter(coords[labels == c, 0], coords[labels == c, 1], label=f"Cluster {c}", alpha=0.6)
                idx = grouped.index[grouped["City"] == city][0]
                plt.scatter(coords[idx, 0], coords[idx, 1], s=120, edgecolors="black", facecolors="none", linewidths=2)
                plt.title(f"Crime Clusters (highlighted: {city})")
                plt.legend()
                plt.tight_layout()

                graph_path = os.path.join(settings.BASE_DIR, "static", "graphs", "cluster_cities.png")
                plt.savefig(graph_path)
                plt.close()
                cluster_plot = "/static/graphs/cluster_cities.png"

    cities = sorted(df["City"].dropna().unique().tolist())
    crimes = sorted(df["Crime Description"].dropna().unique().tolist()[:200])

    return render(request, "cluster.html", {"result": result, "cluster_plot": cluster_plot, "cities": cities, "crimes": crimes})

# ---------------- FUTURE PREDICTION WITH RISK LEVEL ----------------
@login_required(login_url='/login/')
def future_prediction(request):
    result, line_plot, risk_level, risk_color = None, None, None, None
    csv_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")

    if not os.path.exists(csv_path):
        messages.warning(request, "⚠️ Please upload the dataset first.")
        return redirect("/upload/")

    df = pd.read_csv(csv_path, on_bad_lines="skip")

    # Convert dates
    for col in ["Date Reported", "Date of Occurrence"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Date Reported" in df.columns:
        df["year"] = df["Date Reported"].dt.year
    elif "Date of Occurrence" in df.columns:
        df["year"] = df["Date of Occurrence"].dt.year
    else:
        result = "❌ No valid date column found."
        return render(request, "future.html", {"result": result})

    # Dropdown cities
    cities = sorted(df["City"].dropna().unique().tolist())
    selected_city = request.POST.get("city") if request.method == "POST" else None

    if request.method == "POST":
        if selected_city and selected_city != "All":
            df_filtered = df[df["City"] == selected_city]
        else:
            df_filtered = df.copy()

        year_counts = df_filtered.groupby("year").size().reset_index(name="count").dropna()

        if len(year_counts) < 2:
            result = f"⚠️ Not enough data to predict for {selected_city or 'All Cities'}."
        else:
            X = year_counts["year"].values.reshape(-1, 1)
            y = year_counts["count"].values
            model = LinearRegression().fit(X, y)

            next_year = int(year_counts["year"].max()) + 1
            pred = model.predict([[next_year]])[0]

            # 🔥 Risk Level Calculation
            avg = y.mean()
            if pred >= avg * 1.25:
                risk_level, risk_color = "High Risk", "#ff4d4d"  # Red
            elif pred >= avg * 0.75:
                risk_level, risk_color = "Medium Risk", "#ffcc00"  # Yellow
            else:
                risk_level, risk_color = "Low Risk", "#00ff88"  # Green

            result = f"📈 Predicted reported incidents for {selected_city or 'All Cities'} in {next_year}: {pred:.0f} cases."

            # Plot
            plt.figure(figsize=(7, 4))
            plt.plot(year_counts["year"], year_counts["count"], marker="o", label="Past Years", linewidth=2)
            plt.scatter([next_year], [pred], color="red", s=80, label=f"Prediction {next_year}")
            plt.title(f"Crime Trend Prediction - {selected_city or 'All Cities'}", fontsize=12)
            plt.xlabel("Year")
            plt.ylabel("Number of Cases")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            os.makedirs(os.path.join(settings.BASE_DIR, "static", "graphs"), exist_ok=True)
            path = os.path.join(settings.BASE_DIR, "static", "graphs", "future_trend.png")
            plt.savefig(path, dpi=150)
            plt.close()
            line_plot = "/static/graphs/future_trend.png"

    return render(
        request,
        "future.html",
        {
            "result": result,
            "line_plot": line_plot,
            "cities": ["All"] + cities,
            "selected_city": selected_city,
            "risk_level": risk_level,
            "risk_color": risk_color,
        },
    )
