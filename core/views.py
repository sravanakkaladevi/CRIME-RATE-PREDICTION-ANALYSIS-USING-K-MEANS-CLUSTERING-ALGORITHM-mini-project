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
@login_required(login_url='/')
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
# ---------------- CLUSTER PREDICTION (with Elbow Method + India Map) ----------------
@login_required(login_url='/')
def cluster_prediction(request):
    """
    Performs K-Means clustering on the uploaded NCRB dataset.
    - Uses Elbow Method to estimate the number of clusters
    - Generates visual 2D cluster plot
    - Creates interactive India map showing crime zones by city
    """

    result, cluster_plot = None, None

    # --- File setup ---
    csv_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")
    elbow_path = os.path.join(settings.BASE_DIR, "static", "graphs", "elbow_method.png")
    os.makedirs(os.path.dirname(elbow_path), exist_ok=True)

    # --- Dataset check ---
    if not os.path.exists(csv_path):
        messages.warning(request, "⚠️ Please upload the dataset first.")
        return redirect("/upload/")

    # --- Step 1: Load dataset safely ---
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", dtype=str)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df = df.dropna(subset=["City", "Crime Description"])  # must have city & crime
    except Exception as e:
        messages.error(request, f"❌ Error reading dataset: {e}")
        return redirect("/upload/")

    # --- Step 2: Combine text columns for NLP-style clustering ---
    df["combined"] = (
        df["City"].astype(str)
        + " " + df["Crime Description"].astype(str)
        + " " + df["Weapon Used"].astype(str)
        + " " + df["Victim Gender"].astype(str)
    )

    # Group by city (combine all crime text entries per city)
    grouped = df.groupby("City")["combined"].apply(lambda x: " ".join(x)).reset_index()

    # --- Step 3: Convert text data into numerical form using CountVectorizer ---
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans

    vec = CountVectorizer(max_features=2000, stop_words="english")
    X = vec.fit_transform(grouped["combined"])

    # --- Step 4: Find best number of clusters using Elbow Method ---
    distortions = []
    K = range(1, 8)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        distortions.append(km.inertia_)  # total within-cluster variance

    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False

    # Plot the Elbow curve
    plt.figure(figsize=(6, 4))
    if SEABORN_AVAILABLE:
        sns.lineplot(x=list(K), y=distortions, marker='o', color='#00eaff')
    else:
        plt.plot(list(K), distortions, marker='o', color='#00eaff')
    plt.title("Elbow Method - Finding Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(elbow_path)
    plt.close()

    # --- Step 5: Perform K-Means Clustering (3 clusters chosen manually) ---
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    grouped["cluster"] = km.fit_predict(X)

    # --- Step 6: Label clusters based on size (Low, Medium, High) ---
    cluster_counts = grouped.groupby("cluster")["combined"].apply(lambda x: len(x))
    ordered_clusters = cluster_counts.sort_values().index.tolist()
    label_map = {
        ordered_clusters[0]: "Low Crime Zone",
        ordered_clusters[1]: "Medium Crime Zone",
        ordered_clusters[2]: "High Crime Zone",
    }

    # --- Step 7: 🗺️ Create Interactive India Map using Folium ---
    try:
        import folium

        # Base map centered on India
        m = folium.Map(
            location=[22.9734, 78.6569],
            zoom_start=5,
            min_zoom=4,
            max_zoom=7,
            max_bounds=True  # keeps the map locked to India area
        )

        # India boundary box (prevents world panning)
        m.fit_bounds([[6.5, 68.0], [37.1, 97.5]])

        # Predefined city coordinates
        city_coords = {
            "Delhi": [28.6139, 77.2090],
            "Mumbai": [19.0760, 72.8777],
            "Kolkata": [22.5726, 88.3639],
            "Chennai": [13.0827, 80.2707],
            "Hyderabad": [17.3850, 78.4867],
            "Bengaluru": [12.9716, 77.5946],
            "Pune": [18.5204, 73.8567],
            "Ahmedabad": [23.0225, 72.5714],
            "Jaipur": [26.9124, 75.7873],
            "Lucknow": [26.8467, 80.9462],
            "Patna": [25.5941, 85.1376],
            "Bhopal": [23.2599, 77.4126],
            "Visakhapatnam": [17.6868, 83.2185],
            "Vijayawada": [16.5062, 80.6480],
            "Surat": [21.1702, 72.8311],
            "Nagpur": [21.1458, 79.0882],
            "Coimbatore": [11.0168, 76.9558],
            "Indore": [22.7196, 75.8577],
            "Thiruvananthapuram": [8.5241, 76.9366],
            "Chandigarh": [30.7333, 76.7794],
            "Mysuru": [12.2958, 76.6394],
            "Varanasi": [25.3176, 82.9739],
            "Guwahati": [26.1445, 91.7362]
        }

        cluster_colors = {
            "Low Crime Zone": "green",
            "Medium Crime Zone": "orange",
            "High Crime Zone": "red",
        }

        merged = grouped.copy()
        merged["label"] = merged["cluster"].map(label_map)

        for _, row in merged.iterrows():
            city = row["City"]
            if city in city_coords:
                lat, lon = city_coords[city]
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    popup=f"{city} → {row['label']}",
                    color=cluster_colors.get(row["label"], "blue"),
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)

        map_path = os.path.join(settings.BASE_DIR, "static", "graphs", "india_cluster_map.html")
        m.save(map_path)

    except Exception as e:
        print("Map generation skipped:", e)

    # --- Step 8: Handle POST request (user selects city) ---
    result_color = "#ffffff"  # Default color
    result = None  # initialize so it exists for GET requests too

    if request.method == "POST":
        city = request.POST.get("city")
        if city not in grouped["City"].values:
            result = f"⚠️ City '{city}' not found in dataset."
            result_color = "#ffffff"
        else:
            cl = int(grouped.loc[grouped["City"] == city, "cluster"].values[0])
            result = f"{city} is categorized as: {label_map.get(cl, 'Cluster')}"

            # --- Color logic (case-insensitive and fixed colors) ---
            lower_result = result.lower()
            if "low crime" in lower_result:
                result_color = "#00ff88"   # 🟢 Green
            elif "medium crime" in lower_result:
                result_color = "#ffcc00"   # 🟡 Yellow
            elif "high crime" in lower_result:
                result_color = "#ff4d4d"   # 🔴 Red

            # --- 2D cluster visualization ---
            svd = TruncatedSVD(n_components=2, random_state=42)
            coords = svd.fit_transform(X)
            plt.figure(figsize=(6, 4))
            colors = ["#00eaff", "#ffb347", "#a569bd"]
            for c in range(3):
                plt.scatter(coords[grouped["cluster"] == c, 0],
                            coords[grouped["cluster"] == c, 1],
                            label=label_map[c], alpha=0.6)
            idx = grouped.index[grouped["City"] == city][0]
            plt.scatter(coords[idx, 0], coords[idx, 1], s=120,
                        edgecolors="black", facecolors="none", linewidths=2)
            plt.title(f"Crime Clusters (highlighted: {city})")
            plt.legend()
            plt.tight_layout()
            cluster_plot_path = os.path.join(settings.BASE_DIR, "static", "graphs", "cluster_cities.png")
            plt.savefig(cluster_plot_path)
            plt.close()
            cluster_plot = "/static/graphs/cluster_cities.png"

    # --- Step 9: Always render template (for both GET and POST) ---
    cities = sorted(df["City"].dropna().unique().tolist())
    crimes = sorted(df["Crime Description"].dropna().unique().tolist()[:200])
    weapons = sorted(df["Weapon Used"].dropna().unique().tolist()[:50]) if "Weapon Used" in df.columns else []
    genders = sorted(df["Victim Gender"].dropna().unique().tolist()[:50]) if "Victim Gender" in df.columns else []

    return render(
        request,
        "cluster.html",
        {
            "result": result,
            "result_color": result_color,
            "cluster_plot": cluster_plot,
            "elbow_graph": "/static/graphs/elbow_method.png",
            "india_map": "/static/graphs/india_cluster_map.html",
            "cities": cities,
            "crimes": crimes,
            "weapons": weapons,
            "genders": genders,
        },
    )


# ---------------- FUTURE PREDICTION PAGE ----------------
@login_required(login_url='/')
def future_prediction(request):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    import pandas as pd
    import os

    result, line_plot, risk_level, risk_color = None, None, None, None
    csv_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")

    # Check dataset
    if not os.path.exists(csv_path):
        messages.warning(request, "⚠️ Please upload the dataset first.")
        return redirect("/upload/")

    # Load CSV
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", dtype=str)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    except Exception as e:
        messages.error(request, f"❌ Error reading dataset: {e}")
        return redirect("/upload/")

    # Extract year
    for col in ["Date Reported", "Date of Occurrence"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Date Reported" in df.columns and df["Date Reported"].notna().any():
        df["year"] = df["Date Reported"].dt.year
    elif "Date of Occurrence" in df.columns and df["Date of Occurrence"].notna().any():
        df["year"] = df["Date of Occurrence"].dt.year
    else:
        return render(request, "future.html", {"result": "❌ No valid date column found."})

    # Clean and prepare cities list
    df = df.dropna(subset=["City", "year"])
    df["City"] = df["City"].astype(str).str.strip().str.title()
    cities = sorted(set(df["City"].dropna().tolist()))

    selected_city = request.POST.get("city") if request.method == "POST" else None

    if request.method == "POST" and selected_city:
        # handle "All" explicitly
        if selected_city.lower() == "all":
            city_df = df.copy()
        else:
            city_df = df[df["City"].str.lower() == selected_city.lower()]

        if city_df.empty:
            result = f"⚠️ No data found for {selected_city if selected_city.lower() != 'all' else 'overall dataset'}."
        else:
            year_counts = city_df.groupby("year").size().reset_index(name="count").sort_values("year")
            if len(year_counts) < 3:
                result = f"⚠️ Not enough yearly data to predict for {selected_city}."
            else:
                X = year_counts["year"].values.reshape(-1, 1).astype(float)
                y = year_counts["count"].values.astype(float)
                y_smoothed = pd.Series(y).rolling(window=2, min_periods=1).mean().values

                # Pipeline and CV
                pipe = Pipeline([
                    ("poly", PolynomialFeatures(include_bias=False)),
                    ("ridge", Ridge())
                ])
                param_grid = {"poly__degree": [1, 2], "ridge__alpha": [0.01, 0.1, 1.0, 10.0]}
                n_splits = min(4, max(2, len(X) - 1))
                tscv = TimeSeriesSplit(n_splits=n_splits)
                gscv = GridSearchCV(pipe, param_grid, cv=tscv, scoring="r2", n_jobs=1)

                mean_cv_r2, std_cv_r2 = 0.0, 0.0
                try:
                    gscv.fit(X, y_smoothed)
                    best_model = gscv.best_estimator_
                    cv_scores = cross_val_score(best_model, X, y_smoothed, cv=tscv, scoring="r2", n_jobs=1)
                    mean_cv_r2 = float(np.nanmean(cv_scores))
                    std_cv_r2 = float(np.nanstd(cv_scores))
                except Exception:
                    # fallback: fit a simple pipeline (no grid)
                    best_model = pipe.fit(X, y_smoothed)
                    mean_cv_r2, std_cv_r2 = 0.0, 0.0

                # Predict next year
                try:
                    best_model.fit(X, y_smoothed)
                    next_year = int(year_counts["year"].max()) + 1
                    pred_raw = float(best_model.predict(np.array([[next_year]]))[0])
                except Exception:
                    next_year = int(year_counts["year"].max()) + 1
                    pred_raw = float(np.mean(y_smoothed))

                pred = float(max(pred_raw, np.mean(y_smoothed) * 0.5))
                predicted_cases = int(round(pred, 0))

                # Risk Calculation (straightforward, manual)
                trend = y_smoothed[-1] - y_smoothed[-2] if len(y_smoothed) >= 2 else 0.0
                city_mean = float(np.mean(y_smoothed))
                global_mean = float(df.groupby("City").size().mean() if len(df) > 0 else 0.0)
                total_crimes = len(df)
                city_total = len(city_df)
                city_weight = (city_total / total_crimes) if total_crimes > 0 else 0.0

                relative_activity = (city_mean / (global_mean + 1e-6))
                growth_rate = (trend / (abs(city_mean) + 1e-6))
                risk_score = (relative_activity * 2.0) + (growth_rate * 2.5) + (city_weight * 3.0)

                # Metro bias (manual rule)
                if selected_city.lower() in ["delhi", "mumbai", "kolkata", "chennai", "bengaluru", "hyderabad"]:
                    risk_score *= 1.3
                    if predicted_cases >= 200:
                        risk_score += 1.5

                # Thresholds (manual)
                if risk_score >= 4.0:
                    risk_level, risk_color = "High Risk", "#ff4d4d"
                elif risk_score >= 2.0:
                    risk_level, risk_color = "Medium Risk", "#ffcc00"
                else:
                    risk_level, risk_color = "Low Risk", "#00ff88"

                # Safe defaults for CV numbers
                mean_cv_r2 = 0.0 if np.isnan(mean_cv_r2) else mean_cv_r2
                std_cv_r2 = 0.0 if np.isnan(std_cv_r2) else std_cv_r2

                # Train R² (informational)
                try:
                    y_train_pred = best_model.predict(X)
                    train_r2 = r2_score(y_smoothed, y_train_pred)
                except Exception:
                    train_r2 = 0.0

                # Human-friendly confidence label
                if train_r2 >= 0.85:
                    confidence = "Very High"
                elif train_r2 >= 0.65:
                    confidence = "High"
                elif train_r2 >= 0.4:
                    confidence = "Moderate"
                else:
                    confidence = "Low"

                model_accuracy = f"Model confidence: {confidence} (based on {round(train_r2*100,1)}% fit to past data)"

                # Human-readable result
                label = selected_city if selected_city.lower() != "all" else "India (All Cities)"
                result = (
                    f"📈 Predicted total incidents for {label} in {next_year}: "
                    f"{predicted_cases} reported cases — categorized as *{risk_level}*.\n"
                    f"📊 {model_accuracy}"
                )

                # Plot chart
                plt.figure(figsize=(8, 4))
                plt.plot(year_counts["year"], year_counts["count"], marker='o', linewidth=2, label="Observed")
                plt.plot(year_counts["year"], y_smoothed, marker='o', linestyle='--', label="Smoothed")
                plt.scatter([next_year], [predicted_cases], color='red', s=80, label=f"Pred {next_year}")
                plt.title(f"Crime Trend - {label}")
                plt.xlabel("Year")
                plt.ylabel("Number of Crimes")
                plt.legend()
                plt.grid(alpha=0.4, linestyle='--')
                plt.tight_layout()

                graph_dir = os.path.join(settings.BASE_DIR, "static", "graphs")
                os.makedirs(graph_dir, exist_ok=True)
                graph_path = os.path.join(graph_dir, "future_trend.png")
                plt.savefig(graph_path, dpi=150)
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
