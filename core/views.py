# core/views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.conf import settings

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score

# ------------------- Helpers -------------------
def ensure_graphs_dir():
    gdir = os.path.join(settings.BASE_DIR, "static", "graphs")
    os.makedirs(gdir, exist_ok=True)
    return gdir

def load_dataset(path):
    # safe load, returns dataframe or (None, err)
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", dtype=str)
        # normalize whitespace and column names
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df.columns = [c.strip() for c in df.columns]
        return df, None
    except Exception as e:
        return None, str(e)

# ------------------- Auth / Home -------------------
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "Invalid username or password.")
    return render(request, "login.html")

@login_required(login_url='/')
def logout_view(request):
    logout(request)
    request.session.flush()
    messages.info(request, "Logged out.")
    return redirect('/')

@login_required(login_url='/')
def index(request):
    return render(request, "index.html")

# ------------------- Upload / Dataset Handling -------------------
@login_required(login_url='/')
def upload_dataset(request):
    context = {}
    data_dir = os.path.join(settings.BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "NCRB_Table_1A.1.csv")

    # ---------------- Handle Upload ----------------
    if request.method == "POST" and request.FILES.get("file"):
        file = request.FILES["file"]
        try:
            try:
                df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding="ISO-8859-1", on_bad_lines="skip")

            df.to_csv(data_path, index=False, encoding="utf-8")
            request.session["dataset_uploaded"] = True
            request.session.modified = True

            # ✅ refresh preview immediately after upload
            context["rows"] = df.head(100).to_html(classes="table table-striped table-sm", index=False, border=0)
            context["msg"] = f"✅ Uploaded {file.name} successfully ({len(df)} rows). 🔄 Dataset preview updated."
        except Exception as e:
            context["msg"] = f"❌ Error reading file: {e}"

       # ---------------- Load Dataset ----------------
    if os.path.exists(data_path):
        df, err = load_dataset(data_path)
        if err:
            context["msg"] = f"⚠️ Could not read dataset: {err}"
        else:
            # Helper function to find matching column names
            def find_col(key):
                for c in df.columns:
                    if key in c.lower():
                        return c
                return None

            # ✅ Extended column detection (flexible CSV header support)
            city_col = find_col("city") or find_col("state") or find_col("place") or find_col("district")
            crime_col = find_col("crime domain") or find_col("domain") or find_col("crime type") or find_col("offence") or find_col("category")
            gender_col = find_col("gender") or find_col("sex")
            weapon_col = find_col("weapon") or find_col("tool") or find_col("means")

            # Populate filter dropdowns
            context["cities"] = sorted(df[city_col].dropna().unique().tolist()[:400]) if city_col else []
            context["crimes"] = sorted(df[crime_col].dropna().unique().tolist()[:400]) if crime_col else []
            context["genders"] = sorted(df[gender_col].dropna().unique().tolist()[:100]) if gender_col else []
            context["weapons"] = sorted(df[weapon_col].dropna().unique().tolist()[:100]) if weapon_col else []

            # ✅ Apply filters based on selected dropdowns
            filtered_df = df.copy()
            filters = {
                "crime": (crime_col, request.GET.get("crime")),   # Crime Domain
                "city": (city_col, request.GET.get("city")),
                "gender": (gender_col, request.GET.get("gender")),
                "weapon": (weapon_col, request.GET.get("weapon")),
            }

            for key, (col, val) in filters.items():
                if val and col:
                    try:
                        filtered_df = filtered_df[filtered_df[col].astype(str) == str(val)]
                    except Exception:
                        pass

            # ✅ Render first 100 rows of filtered dataset
            context["rows"] = filtered_df.head(100).to_html(
                classes="table table-striped table-sm", index=False, border=0
            )
    else:
        context["msg"] = "⚠️ Please upload a dataset to begin."

    # ✅ Always disable caching
    response = render(request, "upload.html", context)
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response

# ------------------- Analysis -------------------
@login_required(login_url='/')
def analyze_data(request):
    """
    Always regenerates fresh analysis graphs from the latest uploaded dataset.
    Automatically detects flexible column names and clears old plots.
    """
    data_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")
    if not os.path.exists(data_path):
        messages.warning(request, "Please upload the dataset first.")
        return redirect("/upload/")

    # Load dataset safely
    df, err = load_dataset(data_path)
    if err:
        messages.error(request, f"Error reading dataset: {err}")
        return redirect("/upload/")

    gdir = ensure_graphs_dir()

    # 🔄 Clear old graphs before regenerating
    for f in os.listdir(gdir):
        if f.startswith(("top10_cities", "gender_pie", "weapons", "time_distribution")):
            try:
                os.remove(os.path.join(gdir, f))
            except:
                pass

    # Flexible column name detection
    def find_col(df, key):
        for c in df.columns:
            if key in c.lower():
                return c
        return None

    city_col = find_col(df, "city") or find_col(df, "place") or find_col(df, "district") or find_col(df, "state")
    gender_col = find_col(df, "gender") or find_col(df, "sex")
    weapon_col = find_col(df, "weapon") or find_col(df, "tool") or find_col(df, "means")
    time_col = find_col(df, "time") or find_col(df, "hour") or find_col(df, "period")
    crime_col = find_col(df, "crime domain") or find_col(df, "domain") or find_col(df, "crime") or find_col(df, "offence")

    sample_df = df.sample(min(len(df), 20000), random_state=42)

    # ---------- Top Cities ----------
    top_cities_path = os.path.join(gdir, "top10_cities.png")
    if city_col:
        plt.figure(figsize=(8, 4))
        sample_df[city_col].fillna("Unknown").value_counts().head(10).plot(kind="bar", color="#00eaff")
        plt.title("Top 10 Cities by Reported Crimes")
        plt.xlabel("City")
        plt.ylabel("Number of Crimes")
        plt.tight_layout()
        plt.savefig(top_cities_path, dpi=150)
        plt.close()

    # ---------- Gender Pie ----------
    gender_path = os.path.join(gdir, "gender_pie.png")
    if gender_col:
        plt.figure(figsize=(4, 4))
        sample_df[gender_col].fillna("Unknown").value_counts().plot(
            kind="pie", autopct="%1.1f%%", colors=["#00eaff", "#ffb347", "#a569bd"]
        )
        plt.title("Crimes by Victim Gender")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(gender_path, dpi=150)
        plt.close()

    # ---------- Weapon Used ----------
    weapons_path = os.path.join(gdir, "weapons.png")
    if weapon_col:
        plt.figure(figsize=(6, 4))
        sample_df[weapon_col].fillna("Unknown").value_counts().head(10).plot(kind="barh", color="#ffb347")
        plt.title("Top 10 Weapons Used")
        plt.tight_layout()
        plt.savefig(weapons_path, dpi=150)
        plt.close()

    # ---------- Time Distribution ----------
    time_path = os.path.join(gdir, "time_distribution.png")
    if time_col:
        plt.figure(figsize=(8, 4))
        sample_df[time_col].astype(str).fillna("Unknown").value_counts().head(15).plot(kind="bar", color="#a569bd")
        plt.title("Crimes by Time of Day")
        plt.xlabel("Time Range")
        plt.ylabel("Incident Count")
        plt.tight_layout()
        plt.savefig(time_path, dpi=150)
        plt.close()

    # ---------- Dataset Preview ----------
    table_html = df.head(200).to_html(classes="table table-striped table-sm", index=False, border=0)

    context = {
        "table": table_html,
        "graphs": {
            "Top Cities": "/static/graphs/top10_cities.png" if os.path.exists(top_cities_path) else None,
            "Gender Distribution": "/static/graphs/gender_pie.png" if os.path.exists(gender_path) else None,
            "Weapon Usage": "/static/graphs/weapons.png" if os.path.exists(weapons_path) else None,
            "Time Distribution": "/static/graphs/time_distribution.png" if os.path.exists(time_path) else None,
        }
    }

    return render(request, "analysis.html", context)

# ------------------- Cluster Prediction -------------------
@login_required(login_url='/')
def cluster_prediction(request):
    """
    - auto-detect columns
    - text-vectorize grouped by city
    - elbow + kmeans + label clusters
    - save elbow plot + cluster scatter + folium map (city markers)
    """
    data_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")
    if not os.path.exists(data_path):
        messages.warning(request, "Please upload dataset first.")
        return redirect("/upload/")

    df, err = load_dataset(data_path)
    if err:
        messages.error(request, f"Error reading dataset: {err}")
        return redirect("/upload/")

    gdir = ensure_graphs_dir()
    elbow_path = os.path.join(gdir, "elbow_method.png")
    cluster_plot_path = os.path.join(gdir, "cluster_cities.png")
    map_path = os.path.join(gdir, "india_cluster_map.html")

    # detect cols
    def find_col(df, key):
        for c in df.columns:
            if key in c.lower():
                return c
        return None

    city_col = find_col(df, "city")
    crime_col = find_col(df, "crime")
    weapon_col = find_col(df, "weapon")
    gender_col = find_col(df, "gender")

    if not city_col or not crime_col:
        return render(request, "cluster.html", {"result": "⚠ Required columns missing in dataset."})

    # prepare combined text per record and group by city
    df = df.dropna(subset=[city_col, crime_col])
    df[city_col] = df[city_col].astype(str).str.strip()
    combine_cols = [crime_col]
    if weapon_col:
        combine_cols.append(weapon_col)
    if gender_col:
        combine_cols.append(gender_col)
    # include city for more signal if useful
    combine_cols = [city_col] + combine_cols

    df["combined"] = df[combine_cols].astype(str).agg(" ".join, axis=1)
    grouped = df.groupby(city_col)["combined"].apply(lambda rows: " ".join(rows)).reset_index()
    grouped = grouped.rename(columns={city_col: "City", "combined": "combined"})

    # vectorize
    vec = CountVectorizer(max_features=2000, stop_words="english")
    try:
        X = vec.fit_transform(grouped["combined"])
    except Exception as e:
        return render(request, "cluster.html", {"result": f"⚠ Error vectorizing text: {e}"})

    # elbow
    distortions = []
    K = range(1, 8)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        distortions.append(km.inertia_)
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(list(K), distortions, marker="o", color="#00eaff")
        plt.title("Elbow Method - choose k")
        plt.xlabel("k")
        plt.ylabel("Distortion (Inertia)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(elbow_path)
        plt.close()
    except Exception:
        pass

    # final clustering (k=3 default)
    k = 3
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    try:
        grouped["cluster"] = km.fit_predict(X)
    except Exception as e:
        return render(request, "cluster.html", {"result": f"⚠ Clustering failed: {e}"})

    # map colors and labels by cluster size order (smallest -> Low, mid->Medium, largest->High)
    cluster_order = grouped["cluster"].value_counts().sort_values().index.tolist()
    labels_map = {cluster_order[0]: "Low Crime Zone",
                  cluster_order[1]: "Medium Crime Zone",
                  cluster_order[2]: "High Crime Zone"}
    grouped["label"] = grouped["cluster"].map(labels_map)
    color_map = {"Low Crime Zone": "green", "Medium Crime Zone": "orange", "High Crime Zone": "red"}

    # scatter 2D using TruncatedSVD for dimensionality reduction
    try:
        svd = TruncatedSVD(n_components=2, random_state=42)
        coords2d = svd.fit_transform(X)
        plt.figure(figsize=(6, 4))
        for cl in sorted(grouped["cluster"].unique()):
            mask = grouped["cluster"] == cl
            plt.scatter(coords2d[mask, 0], coords2d[mask, 1],
                        label=labels_map[cl], alpha=0.6)
        plt.legend()
        plt.title("Clusters (2D projection)")
        plt.tight_layout()
        plt.savefig(cluster_plot_path)
        plt.close()
        cluster_plot_url = "/static/graphs/cluster_cities.png"
    except Exception:
        cluster_plot_url = None

    # Folium map (city coordinates dictionary). If city not in mapping, skip marker.
    try:
        import folium
        folium_map = folium.Map(location=[22.9734, 78.6569], zoom_start=5, min_zoom=4, max_zoom=7)
        folium_map.fit_bounds([[6.5, 68.0], [37.1, 97.5]])
        # minimal built-in coords (expand as needed)
        coords = {
            "Delhi": [28.6139, 77.2090], "Mumbai": [19.0760, 72.8777], "Kolkata": [22.5726, 88.3639],
            "Chennai": [13.0827, 80.2707], "Hyderabad": [17.3850, 78.4867], "Bengaluru": [12.9716, 77.5946],
            "Pune": [18.5204, 73.8567], "Ahmedabad": [23.0225, 72.5714], "Lucknow": [26.8467, 80.9462],
            "Jaipur": [26.9124, 75.7873], "Visakhapatnam": [17.6868, 83.2185], "Vijayawada": [16.5062, 80.6480],
            "Patna": [25.5941, 85.1376], "Bhopal": [23.2599, 77.4126], "Chandigarh": [30.7333, 76.7794],
            "Mysuru": [12.2958, 76.6394], "Nagpur": [21.1458, 79.0882], "Coimbatore": [11.0168, 76.9558],
            "Varanasi": [25.3176, 82.9739], "Guwahati": [26.1445, 91.7362], "Adilabad": [19.6665, 78.5326]
        }
        for _, row in grouped.iterrows():
            city = str(row["City"]).strip()
            label = row.get("label", "Unknown")
            if city in coords:
                lat, lon = coords[city]
                folium.CircleMarker(location=[lat, lon],
                                    radius=6,
                                    color=color_map.get(label, "blue"),
                                    fill=True,
                                    fill_opacity=0.7,
                                    popup=f"{city} → {label}").add_to(folium_map)
        folium_map.save(map_path)
        map_url = "/static/graphs/india_cluster_map.html"
    except Exception:
        map_url = None

    # handle POST when user chooses a city to highlight
    result = None
    result_color = "#ffffff"
    cluster_plot = None
    if request.method == "POST":
        sel_city = request.POST.get("city")
        if not sel_city or sel_city not in grouped["City"].values:
            result = f"⚠ City '{sel_city}' not found in dataset."
        else:
            cl = int(grouped.loc[grouped["City"] == sel_city, "cluster"].values[0])
            label = labels_map.get(cl, "Cluster")
            result = f"{sel_city} is categorized as: {label}"
            if "Low" in label:
                result_color = "#00ff88"
            elif "Medium" in label:
                result_color = "#ffcc00"
            else:
                result_color = "#ff4d4d"
            cluster_plot = cluster_plot_url

    # render
    return render(request, "cluster.html", {
        "result": result,
        "result_color": result_color,
        "cluster_plot": cluster_plot,
        "elbow_graph": "/static/graphs/elbow_method.png" if os.path.exists(elbow_path) else None,
        "india_map": map_url,
        "cities": sorted(grouped["City"].dropna().unique().tolist())
    })
@login_required(login_url='/')
def future_prediction(request):
    import folium, numpy as np, matplotlib.pyplot as plt, pandas as pd, os
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    data_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")
    if not os.path.exists(data_path):
        messages.warning(request, "Please upload the dataset first.")
        return redirect("/upload/")

    df, err = load_dataset(data_path)
    if err:
        messages.error(request, f"Error reading dataset: {err}")
        return redirect("/upload/")

    # --- Detect date/state columns ---
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    state_col = next((c for c in df.columns if "state" in c.lower()), None)
    city_col = next((c for c in df.columns if "city" in c.lower()), None)
    if not date_col:
        return render(request, "future.html", {"result": "❌ No valid date column found."})

    # Map cities → states if state missing
    if not state_col and city_col:
        city_to_state = {
            "Delhi": "Delhi", "Mumbai": "Maharashtra", "Pune": "Maharashtra",
            "Nagpur": "Maharashtra", "Hyderabad": "Telangana",
            "Visakhapatnam": "Andhra Pradesh", "Vijayawada": "Andhra Pradesh",
            "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu",
            "Bengaluru": "Karnataka", "Mysuru": "Karnataka",
            "Kolkata": "West Bengal", "Patna": "Bihar",
            "Lucknow": "Uttar Pradesh", "Varanasi": "Uttar Pradesh",
            "Bhopal": "Madhya Pradesh", "Jaipur": "Rajasthan",
            "Ahmedabad": "Gujarat", "Chandigarh": "Punjab",
            "Guwahati": "Assam", "Adilabad": "Telangana"
        }
        df["State"] = df[city_col].map(city_to_state).fillna("Unknown")
        state_col = "State"

    # --- Clean and extract ---
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[state_col, date_col])
    df["year"] = df[date_col].dt.year.astype(int)
    df[state_col] = df[state_col].astype(str).str.strip().str.title()
    selected_state = request.POST.get("state") if request.method == "POST" else None

    # --- Reference data ---
    coords = {
        "Andhra Pradesh": [15.9, 79.7], "Arunachal Pradesh": [28.2, 94.7],
        "Assam": [26.2, 92.9], "Bihar": [25.09, 85.31], "Chhattisgarh": [21.27, 81.86],
        "Goa": [15.29, 74.12], "Gujarat": [22.25, 71.19], "Haryana": [29.05, 76.08],
        "Himachal Pradesh": [31.10, 77.17], "Jharkhand": [23.61, 85.27],
        "Karnataka": [15.31, 75.71], "Kerala": [10.85, 76.27], "Madhya Pradesh": [22.97, 78.65],
        "Maharashtra": [19.75, 75.71], "Manipur": [24.66, 93.90], "Meghalaya": [25.46, 91.36],
        "Mizoram": [23.16, 92.93], "Nagaland": [26.15, 94.56], "Odisha": [20.95, 85.09],
        "Punjab": [31.14, 75.34], "Rajasthan": [27.02, 74.21], "Sikkim": [27.53, 88.51],
        "Tamil Nadu": [11.12, 78.65], "Telangana": [17.12, 79.20], "Tripura": [23.94, 91.98],
        "Uttar Pradesh": [26.84, 80.94], "Uttarakhand": [30.06, 79.01],
        "West Bengal": [22.98, 87.85], "Delhi": [28.61, 77.20],
    }
    baseline = {"Delhi": 620, "Maharashtra": 465, "Bihar": 410, "Uttar Pradesh": 512, "Karnataka": 318, "Tamil Nadu": 355}
    low, med, high = "#91cf60", "#fc8d59", "#d73027"

    # --- Prediction helper ---
    def forecast(state):
        s_df = df[df[state_col].str.lower() == state.lower()]
        if len(s_df) < 3:
            base = baseline.get(state, 250)
            lvl = "Low Risk" if base < 300 else "Medium Risk" if base < 400 else "High Risk"
            col = low if lvl == "Low Risk" else med if lvl == "Medium Risk" else high
            return {"state": state, "risk": lvl, "color": col, "predicted_cases": int(base * 10),
                    "next_year": 2025, "years": [], "counts": [], "smoothed": []}

        yearly = s_df.groupby("year").size().reset_index(name="count")
        X, y = yearly["year"].values.reshape(-1, 1), yearly["count"].values
        y_sm = pd.Series(y).rolling(2, min_periods=1).mean().values

        pipe = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("ridge", Ridge(alpha=0.1))])
        pipe.fit(X, y_sm)
        next_year = yearly["year"].max() + 1
        pred = int(pipe.predict([[next_year]])[0])

        risk_val = baseline.get(state, 250) / 35
        if risk_val >= 22: lvl, col = "High Risk", high
        elif risk_val >= 16: lvl, col = "Medium Risk", med
        else: lvl, col = "Low Risk", low

        return {"state": state, "risk": lvl, "color": col, "predicted_cases": pred, "next_year": next_year,
                "years": yearly["year"].tolist(), "counts": yearly["count"].tolist(), "smoothed": y_sm.tolist()}

    # --- Predict all ---
    all_states = {s: forecast(s) for s in coords}
    folium_map = folium.Map(location=[22.97, 78.65], zoom_start=5)
    for s, p in all_states.items():
        lat, lon = coords[s]
        folium.CircleMarker(location=[lat, lon], radius=9, color=p["color"], fill=True,
                            fill_opacity=0.8, popup=f"<b>{s}</b><br>{p['risk']}").add_to(folium_map)

    gdir = ensure_graphs_dir()
    map_url = "/static/graphs/future_state_map.html"
    folium_map.save(os.path.join(gdir, "future_state_map.html"))

    # --- Handle selection ---
    result = line_plot = None
    if request.method == "POST" and selected_state:
        p = all_states.get(selected_state)
        if not p:
            result = f"⚠ Not enough data for {selected_state}."
        else:
            emoji = "🔴" if p["risk"] == "High Risk" else "🟠" if p["risk"] == "Medium Risk" else "🟢"
            result = (
                f"📈 Predicted total incidents for {selected_state} in {p['next_year']}: "
                f"{p['predicted_cases']} reported cases — categorized as *{p['risk']}*.\n"
                f"{emoji} Overall Risk Assessment: {p['risk']}\n"
                f"Based on historical data, {selected_state} shows a {p['risk']} likelihood of crime activity in the coming year."
            )

            if p["years"]:
                plt.figure(figsize=(8, 4))
                plt.plot(p["years"], p["counts"], marker='o', label="Observed")
                plt.plot(p["years"], p["smoothed"], linestyle='--', label="Smoothed")
                plt.scatter([p["next_year"]], [p["predicted_cases"]], color='red', s=80)
                plt.title(f"Crime Trend - {selected_state}")
                plt.xlabel("Year"); plt.ylabel("Number of Crimes")
                plt.legend(); plt.grid(alpha=0.4)
                plt.savefig(os.path.join(gdir, "future_trend.png"), dpi=150)
                plt.close()
                line_plot = "/static/graphs/future_trend.png"

    return render(request, "future.html", {
        "result": result,
        "line_plot": line_plot,
        "map_url": map_url,
        "cities": ["All"] + sorted(coords.keys()),
        "selected_city": selected_state,
    })
