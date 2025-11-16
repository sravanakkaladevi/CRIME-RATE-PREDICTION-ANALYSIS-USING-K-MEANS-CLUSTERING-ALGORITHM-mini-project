# core/views.py
from django.http import HttpResponse
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
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score

# ------------------- Helpers -------------------
def delete_if_exists(path):
    import os
    if os.path.exists(path):
        os.remove(path)

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

    data_path = os.path.join(data_dir, "crime_dataset.csv")
        # ------------------ REFRESH PREVIEW ------------------
    if request.GET.get("refresh") == "1":
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            context["rows"] = df.head(100).to_html(
                classes="table table-striped table-sm",
                index=False,
                border=0
            )
            context["msg"] = "🔄 Dataset preview refreshed."
        return render(request, "upload.html", context)

# ------------------ SHOW SAVED DATASET ------------------
@login_required(login_url='/')
def upload_dataset(request):
    context = {}
    data_dir = os.path.join(settings.BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    data_path = os.path.join(data_dir, "crime_dataset.csv")

    # ------------------ FILE UPLOAD ------------------
    if request.method == "POST" and request.FILES.get("file"):
        file = request.FILES["file"]

        try:
            df = pd.read_csv(file)

            required_cols = [
                "State","Year","Population","Total_Crimes","Crime_Rate",
                "Murder","Rape","Kidnapping","Assault","Hurt",
                "Robbery","Theft","House_Breaking","Cyber_Crime"
            ]

            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                context["msg"] = f"❌ Missing required columns: {missing}"
            else:
                df.to_csv(data_path, index=False, encoding="utf-8")
                context["msg"] = f"✅ Uploaded {file.name} successfully ({len(df)} rows)."

        except Exception as e:
            context["msg"] = f"❌ Error reading file: {e}"

    # ------------------ REFRESH PREVIEW ------------------
    if request.GET.get("refresh") == "1":
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            context["rows"] = df.head(100).to_html(
                classes="table table-striped table-sm",
                index=False,
                border=0
            )
            context["msg"] = "🔄 Dataset preview refreshed."
        else:
            context["msg"] = "⚠️ No dataset to refresh."
            context["rows"] = None
        return render(request, "upload.html", context)

    # ------------------ SHOW SAVED DATASET ------------------
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        states = sorted(df["State"].unique())
        years = sorted(df["Year"].unique())
        crime_columns = [
            "Crime_Rate","Murder","Rape","Kidnapping","Assault","Hurt",
            "Robbery","Theft","House_Breaking","Cyber_Crime"
        ]

        state = request.GET.get("state")
        year = request.GET.get("year")
        crime = request.GET.get("crime")

        filtered_df = df.copy()

        if state:
            filtered_df = filtered_df[filtered_df["State"] == state]

        if year:
            try:
                filtered_df = filtered_df[filtered_df["Year"] == int(year)]
            except:
                pass

        if crime and crime in crime_columns:
            filtered_df = filtered_df.sort_values(crime, ascending=False)

        context["rows"] = filtered_df.head(100).to_html(
            classes="table table-striped table-sm",
            index=False,
            border=0
        )

        context["msg"] = f"📄 Loaded dataset ({len(df)} rows)."
        context["states"] = states
        context["years"] = years
        context["crime_columns"] = crime_columns

    else:
        context["msg"] = "⚠️ No dataset uploaded yet."
        context["rows"] = None

    # ------------------ FINAL RETURN ------------------
    response = render(request, "upload.html", context)
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response
# ------------------- Data Analysis -------------------
@login_required(login_url='/')
def analyze_data(request):
    import traceback
    import folium, json

    try:
        # Load dataset
        data_path = os.path.join(settings.BASE_DIR, "data", "crime_dataset.csv")
        if not os.path.exists(data_path):
            messages.warning(request, "Please upload the dataset first.")
            return redirect("/upload/")

        df = pd.read_csv(data_path)

        # Graph directory
        gdir = ensure_graphs_dir()

        # Remove old graphs if exist
        for f in os.listdir(gdir):
            if f.startswith(("crime_rate_by_state",
                             "total_crimes_year",
                             "cyber_crime_state",
                             "crime_heatmap",
                             "choropleth_map")):
                try:
                    os.remove(os.path.join(gdir, f))
                except:
                    pass

        # ----------------------------------------------------------
        # CLEAN & NORMALIZE STATE NAMES (MATCH WITH GEOJSON EXACTLY)
        # ----------------------------------------------------------
        df["Year"] = df["Year"].astype(int)
        df["State"] = df["State"].astype(str).str.strip()

        # Full correct mapping (DF → GeoJSON)
        state_fix = {
            "Andaman & Nicobar Islands": "Andaman and Nicobar",

            "Jammu & Kashmir": "Jammu and Kashmir",
            "Jammu And Kashmir": "Jammu and Kashmir",
            "Ladakh": "Jammu and Kashmir",  # Not present in your GeoJSON

            "NCT of Delhi": "Delhi",

            "Odisha": "Orissa",

            "Uttarakhand": "Uttaranchal",

            "Pondicherry": "Puducherry",

            "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
            "Daman & Diu": "Daman and Diu",

            "Chattisgarh": "Chhattisgarh",
        }

        df["State"] = df["State"].replace(state_fix)

        # ----------------------------------------------------------
        # CRIME LEVEL CATEGORIES
        # ----------------------------------------------------------
        df_state_total = df.groupby("State")["Total_Crimes"].sum().reset_index()
        df_state_total = df_state_total.sort_values("Total_Crimes", ascending=False)
        n = len(df_state_total)

        crime_levels = {
            "high": df_state_total.head(n // 3)["State"].tolist(),
            "medium": df_state_total.iloc[n // 3: 2 * n // 3]["State"].tolist(),
            "low": df_state_total.tail(n // 3)["State"].tolist(),
        }

        # ----------------------------------------------------------
        # GRAPH 1 — Crime Rate by State
        # ----------------------------------------------------------
        plt.figure(figsize=(12, 5))
        df.groupby("State")["Crime_Rate"].mean().sort_values().plot(kind="bar")
        plt.tight_layout()
        plt.title("Average Crime Rate by State")
        plt.savefig(os.path.join(gdir, "crime_rate_by_state.png"))
        plt.close()

        # ----------------------------------------------------------
        # GRAPH 2 — Total Crimes per Year
        # ----------------------------------------------------------
        plt.figure(figsize=(8, 5))
        df.groupby("Year")["Total_Crimes"].sum().plot(marker="o")
        plt.title("India Total Crimes (2020–2024)")
        plt.tight_layout()
        plt.savefig(os.path.join(gdir, "total_crimes_year.png"))
        plt.close()

        # ----------------------------------------------------------
        # GRAPH 3 — Cyber Crime by State
        # ----------------------------------------------------------
        plt.figure(figsize=(12, 5))
        df.groupby("State")["Cyber_Crime"].mean().sort_values().plot(
            kind="bar",
            color="orange"
        )
        plt.title("Cyber Crime Hotspots")
        plt.tight_layout()
        plt.savefig(os.path.join(gdir, "cyber_crime_state.png"))
        plt.close()

        # ----------------------------------------------------------
        # STATIC HEATMAP (PNG)
        # ----------------------------------------------------------
        heatmap_url = None
        try:
            heatmap_path = os.path.join(gdir, "crime_heatmap.png")
            pivot = df.groupby("State")["Total_Crimes"].sum().sort_values()

            plt.figure(figsize=(12, 10))
            plt.barh(pivot.index, pivot.values, color=plt.cm.hot(pivot.values / pivot.max()))
            plt.tight_layout()
            plt.title("Crime Density Heatmap")
            plt.savefig(heatmap_path, dpi=150)
            plt.close()

            heatmap_url = "/static/graphs/crime_heatmap.png"
        except:
            print("Heatmap error")

        # ----------------------------------------------------------
        # INTERACTIVE MAP (FOLIUM CHOROPLETH)
        # ----------------------------------------------------------
        choropleth_url = None
        try:
            geojson_path = os.path.join(settings.BASE_DIR, "static", "geojson", "india_states.geojson")

            with open(geojson_path, "r", encoding="utf-8") as f:
                india_geojson = json.load(f)

            crime_data = df.groupby("State")["Total_Crimes"].sum().reset_index()
            crime_dict = dict(zip(crime_data["State"], crime_data["Total_Crimes"]))

            # Add crime data inside GeoJSON
            for feature in india_geojson["features"]:
                nm = feature["properties"]["NAME_1"]
                feature["properties"]["TOTAL_CRIME"] = int(crime_dict.get(nm, 0))

            m = folium.Map(location=[22.97, 78.65], zoom_start=5)

            # Choropleth layer
            folium.Choropleth(
                geo_data=geojson_path,
                name="choropleth",
                data=crime_data,
                columns=["State", "Total_Crimes"],
                key_on="feature.properties.NAME_1",
                fill_color="YlOrRd",
                fill_opacity=0.9,
                legend_name="Crime Rate by State"
            ).add_to(m)

            # Tooltip Layer
            folium.GeoJson(
                india_geojson,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["NAME_1", "TOTAL_CRIME"],
                    aliases=["State:", "Total Crimes:"],
                ),
                style_function=lambda x: {"color": "transparent", "weight": 0},
                highlight_function=lambda x: {"weight": 3, "color": "black"},
            ).add_to(m)

            # ----------------------------------------------------------
            # FIXED ZOOM FUNCTION — WORKS FOR DELHI + ALL UTs
            # ----------------------------------------------------------
            if request.method == "POST":
                sel = request.POST.get("state")
                if sel:
                    for feat in india_geojson["features"]:
                        if feat["properties"]["NAME_1"] == sel:
                            gj = folium.GeoJson(feat)
                            gj.add_to(m)
                            m.fit_bounds(gj.get_bounds())
                            break

            # Save map
            choropleth_path = os.path.join(gdir, "choropleth_map.html")
            m.save(choropleth_path)
            choropleth_url = "/static/graphs/choropleth_map.html"

        except Exception as e:
            print("Map error:", e)
            traceback.print_exc()

        # ----------------------------------------------------------
        # DATA TABLE
        # ----------------------------------------------------------
        table_html = df.head(200).to_html(
            classes="table table-striped table-sm",
            index=False
        )

        # ----------------------------------------------------------
        # SEND TO TEMPLATE
        # ----------------------------------------------------------
        context = {
            "table": table_html,
            "graphs": {
                "Crime Rate by State": "/static/graphs/crime_rate_by_state.png",
                "Total Crimes by Year": "/static/graphs/total_crimes_year.png",
                "Cyber Crime by State": "/static/graphs/cyber_crime_state.png",
            },
            "crime_levels": crime_levels,
            "heatmap": heatmap_url,
            "choropleth": choropleth_url,
        }

        return render(request, "analysis.html", context)

    except Exception as e:
        print("FATAL ERROR:", e)
        traceback.print_exc()
        return HttpResponse("Analysis failed", status=500)


# ------------------- Cluster Prediction -------------------
@login_required(login_url='/')
def cluster_prediction(request):
    import folium
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    # Dataset path
    data_path = os.path.join(settings.BASE_DIR, "data", "crime_dataset.csv")
    if not os.path.exists(data_path):
        messages.warning(request, "Please upload the dataset first.")
        return redirect("/upload/")

    df = pd.read_csv(data_path)
    df["State"] = df["State"].astype(str).str.strip()
    
    # Convert numeric columns
    numeric_cols = ["Crime_Rate", "Murder", "Rape", "Kidnapping", "Assault", "Hurt", 
                   "Robbery", "Theft", "House_Breaking", "Cyber_Crime", "Total_Crimes"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    gdir = ensure_graphs_dir()
    elbow_path = os.path.join(gdir, "elbow_method.png")
    cluster_plot_path = os.path.join(gdir, "cluster_states.png")
    map_path = os.path.join(gdir, "india_cluster_map.html")

    # ---------------- Prepare Data ----------------
    grouped = df.groupby("State")[numeric_cols].mean().reset_index()
    X = grouped[["Crime_Rate", "Murder", "Theft", "Cyber_Crime"]].fillna(0)

    # ---------------- Elbow Plot ----------------
    distortions = []
    for k in range(1, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 7), distortions, marker="o")
    plt.title("Elbow Method - Choose k")
    plt.savefig(elbow_path)
    plt.close()

    # ---------------- Final KMeans ----------------
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    grouped["cluster"] = km.fit_predict(X)

    avg_rate = grouped.groupby("cluster")["Crime_Rate"].mean().sort_values()
    label_map = {
        avg_rate.index[0]: "Low Crime Zone",
        avg_rate.index[1]: "Medium Crime Zone",
        avg_rate.index[2]: "High Crime Zone",
    }
    grouped["label"] = grouped["cluster"].map(label_map)

    # ---------------- Scatter Plot (SVD Dimensionality) ----------------
    svd = TruncatedSVD(n_components=2)
    coords2d = svd.fit_transform(X)

    plt.figure(figsize=(6, 4))
    for cl in grouped["cluster"].unique():
        mask = grouped["cluster"] == cl
        plt.scatter(coords2d[mask, 0], coords2d[mask, 1], label=label_map[cl])
    plt.legend()
    plt.savefig(cluster_plot_path)
    plt.close()

    # ---------------- Coordinates ----------------
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
        "Andaman & Nicobar Islands": [11.7401, 92.6586]
    }

    # ---------------- Zoom Logic ----------------
    selected_state = request.POST.get("state")
    map_center = [22.97, 78.65]
    map_zoom = 5

    if selected_state and selected_state in coords:
        map_center = coords[selected_state]
        map_zoom = 7

    # ---------------- Create Folium Map ----------------
    folium_map = folium.Map(location=map_center, zoom_start=map_zoom)
    color_map = {"Low Crime Zone": "green", "Medium Crime Zone": "orange", "High Crime Zone": "red"}

    for _, row in grouped.iterrows():
        st = row["State"]
        if st in coords:
            folium.CircleMarker(
                location=coords[st],
                radius=8,
                color=color_map[row["label"]],
                fill=True,
                fill_opacity=0.8,
                tooltip=f"{st} → {row['label']}"
            ).add_to(folium_map)

    delete_if_exists(map_path)
    folium_map.save(map_path)

    # ---------------- Response ----------------
    result = None
    result_color = "#ffffff"

    if selected_state and selected_state in grouped["State"].values:
        label = grouped.loc[grouped["State"] == selected_state, "label"].values[0]
        result = f"{selected_state} → {label}"
        result_color = "#00ff66" if "Low" in label else "#ffaa33" if "Medium" in label else "#ff4444"

    return render(request, "cluster.html", {
        "elbow_graph": "/static/graphs/elbow_method.png",
        "cluster_plot": "/static/graphs/cluster_states.png",
        "india_map": "/static/graphs/india_cluster_map.html",
        "states": sorted(grouped["State"].tolist()),
        "result": result,
        "result_color": result_color,
        "selected_state": selected_state
    })


# ------------------- Future Prediction -------------------
@login_required(login_url='/')
def future_prediction(request):
    import folium
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures

    data_path = os.path.join(settings.BASE_DIR, "data", "crime_dataset.csv")
    if not os.path.exists(data_path):
        messages.warning(request, "Upload dataset first.")
        return redirect("/upload/")

    df = pd.read_csv(data_path)
    df["Year"] = df["Year"].astype(int)
    df["State"] = df["State"].astype(str)

    states = sorted(df["State"].unique())
    selected_state = request.POST.get("state")

    gdir = ensure_graphs_dir()

    # ---------------- Coordinates ----------------
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
        "Andaman & Nicobar Islands": [11.7401, 92.6586]
    }

    # ---------------- Latest Year ----------------
    latest_df = df.sort_values("Year").groupby("State").tail(1)

    kmeans = KMeans(n_clusters=3, random_state=42)
    latest_df["Cluster"] = kmeans.fit_predict(latest_df[["Total_Crimes"]])

    cluster_order = latest_df.groupby("Cluster")["Total_Crimes"].mean().sort_values().index.tolist()
    cluster_to_risk = {
        cluster_order[0]: "Low Risk",
        cluster_order[1]: "Medium Risk",
        cluster_order[2]: "High Risk"
    }
    cluster_to_color = {
        cluster_order[0]: "green",
        cluster_order[1]: "yellow",
        cluster_order[2]: "red"
    }

    # ---------------- Zoom Logic ----------------
    map_center = [22.97, 78.65]
    map_zoom = 5

    if selected_state and selected_state in coords:
        map_center = coords[selected_state]
        map_zoom = 7

    # ---------------- Create Map ----------------
    folium_map = folium.Map(location=map_center, zoom_start=map_zoom)

    for _, row in latest_df.iterrows():
        st = row["State"]
        if st in coords:
            folium.CircleMarker(
                location=coords[st],
                radius=7,
                color=cluster_to_color[row["Cluster"]],
                fill=True,
                fill_opacity=0.8,
                tooltip=f"{st} - Crimes: {row['Total_Crimes']}"
            ).add_to(folium_map)

    map_path = os.path.join(gdir, "future_india_map.html")
    delete_if_exists(map_path)
    folium_map.save(map_path)

    # ---------------- Forecast ----------------
    result = None
    risk_level = None
    line_plot = None

    if selected_state:
        s_df = df[df["State"] == selected_state]
        yearly = s_df.groupby("Year")["Total_Crimes"].sum().reset_index()

        if len(yearly) >= 3:
            X = yearly["Year"].values.reshape(-1, 1)
            y = yearly["Total_Crimes"].values

            poly = PolynomialFeatures(degree=2)
            Xp = poly.fit_transform(X)

            model = Ridge(alpha=0.1)
            model.fit(Xp, y)

            years = np.array([2025, 2026]).reshape(-1, 1)
            preds = model.predict(poly.transform(years))

            # Plot
            plt.figure(figsize=(8, 4))
            plt.plot(yearly["Year"], y, marker="o", label="Observed")
            plt.plot([2025, 2026], preds, marker="x", label="Predicted")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(gdir, "future_trend.png"))
            plt.close()

            line_plot = "/static/graphs/future_trend.png"
            result = f"{selected_state}: 2025 → {int(preds[0])} | 2026 → {int(preds[1])}"
        else:
            result = f"{selected_state}: Not enough data for prediction."

        cluster_id = latest_df.loc[latest_df["State"] == selected_state, "Cluster"].values[0]
        risk_level = cluster_to_risk[cluster_id]

    return render(request, "future.html", {
        "states": states,
        "selected_state": selected_state,
        "result": result,
        "risk_level": risk_level,
        "line_plot": line_plot,
        "map_url": "/static/graphs/future_india_map.html",
    })


# ------------------- Advanced Analysis (Pending & Solved Cases) -------------------
@login_required(login_url='/')
def advanced_analysis(request):
    import traceback
    import json
    
    try:
        # Load dataset
        data_path = os.path.join(settings.BASE_DIR, "data", "crime_dataset.csv")
        if not os.path.exists(data_path):
            messages.warning(request, "Please upload the dataset first.")
            return redirect("/upload/")

        df = pd.read_csv(data_path)
        df["Year"] = df["Year"].astype(int)
        df["State"] = df["State"].astype(str).str.strip()
        
        # Parse Pending_Percentage and Crime_Solved if needed
        if "Pending_Percentage" in df.columns:
            df["Pending_Percentage"] = df["Pending_Percentage"].astype(str).str.replace("%", "").astype(float)
        
        if "Crime_Solved" in df.columns:
            df["Crime_Solved"] = pd.to_numeric(df["Crime_Solved"], errors='coerce')

        gdir = ensure_graphs_dir()

        # ----------------------------------------------------------
        # ANALYSIS 1: Crime Solved Rate by State
        # ----------------------------------------------------------
        plt.figure(figsize=(12, 6))
        if "Crime_Solved" in df.columns and "Total_Crimes" in df.columns:
            df_analysis = df.groupby("State").agg({
                "Crime_Solved": "sum",
                "Total_Crimes": "sum"
            }).reset_index()
            df_analysis["Solved_Rate"] = (df_analysis["Crime_Solved"] / df_analysis["Total_Crimes"] * 100).round(2)
            df_analysis = df_analysis.sort_values("Solved_Rate", ascending=True)
            
            plt.barh(df_analysis["State"], df_analysis["Solved_Rate"], color="steelblue")
            plt.xlabel("Crime Solved Rate (%)")
            plt.title("Crime Resolution Rate by State (2020-2024)")
            plt.tight_layout()
            plt.savefig(os.path.join(gdir, "crime_solved_rate.png"), dpi=100)
            plt.close()

        # ----------------------------------------------------------
        # ANALYSIS 2: Pending Cases Analysis
        # ----------------------------------------------------------
        plt.figure(figsize=(12, 6))
        if "Pending_Percentage" in df.columns:
            pending_by_state = df.groupby("State")["Pending_Percentage"].mean().sort_values(ascending=False).head(15)
            
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(pending_by_state)))
            plt.barh(pending_by_state.index, pending_by_state.values, color=colors)
            plt.xlabel("Average Pending Percentage (%)")
            plt.title("Top 15 States with Highest Pending Cases")
            plt.tight_layout()
            plt.savefig(os.path.join(gdir, "pending_cases.png"), dpi=100)
            plt.close()

        # ----------------------------------------------------------
        # ANALYSIS 3: Trend Over Years (Solved vs Pending)
        # ----------------------------------------------------------
        plt.figure(figsize=(10, 6))
        if "Crime_Solved" in df.columns and "Pending_Percentage" in df.columns:
            yearly_stats = df.groupby("Year").agg({
                "Crime_Solved": "sum",
                "Total_Crimes": "sum",
                "Pending_Percentage": "mean"
            }).reset_index()
            yearly_stats["Solved_Rate"] = (yearly_stats["Crime_Solved"] / yearly_stats["Total_Crimes"] * 100).round(2)
            
            ax1 = plt.subplot(111)
            ax1.plot(yearly_stats["Year"], yearly_stats["Solved_Rate"], marker="o", label="Solved Rate (%)", color="green", linewidth=2)
            ax1.set_ylabel("Crime Solved Rate (%)", color="green")
            ax1.tick_params(axis="y", labelcolor="green")
            
            ax2 = ax1.twinx()
            ax2.plot(yearly_stats["Year"], yearly_stats["Pending_Percentage"], marker="s", label="Avg Pending (%)", color="red", linewidth=2)
            ax2.set_ylabel("Average Pending Percentage (%)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            
            plt.title("Crime Solved vs Pending Cases Trend (2020-2024)")
            ax1.set_xlabel("Year")
            plt.tight_layout()
            plt.savefig(os.path.join(gdir, "solved_vs_pending_trend.png"), dpi=100)
            plt.close()

        # ----------------------------------------------------------
        # ANALYSIS 4: Correlation Matrix
        # ----------------------------------------------------------
        plt.figure(figsize=(8, 6))
        numeric_cols = ["Crime_Rate", "Murder", "Rape", "Theft", "Cyber_Crime"]
        if "Pending_Percentage" in df.columns:
            numeric_cols.append("Pending_Percentage")
        if "Crime_Solved" in df.columns:
            numeric_cols.append("Crime_Solved")
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            import seaborn as sns
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
            plt.title("Correlation Matrix: Crime Variables")
            plt.tight_layout()
            plt.savefig(os.path.join(gdir, "correlation_matrix.png"), dpi=100)
            plt.close()

        # ----------------------------------------------------------
        # ANALYSIS 5: Top Performers (Highest Crime Solved Rate)
        # ----------------------------------------------------------
        if "Crime_Solved" in df.columns:
            top_performers = df_analysis.nlargest(10, "Solved_Rate")[["State", "Solved_Rate"]]
            
            plt.figure(figsize=(10, 6))
            plt.barh(top_performers["State"], top_performers["Solved_Rate"], color="green", alpha=0.7)
            plt.xlabel("Crime Solved Rate (%)")
            plt.title("Top 10 Performers: Highest Crime Resolution Rate")
            plt.tight_layout()
            plt.savefig(os.path.join(gdir, "top_performers.png"), dpi=100)
            plt.close()

        # ----------------------------------------------------------
        # ANALYSIS 6: Statistics Summary
        # ----------------------------------------------------------
        stats_data = {
            "Total States": df["State"].nunique(),
            "Years Covered": f"{df['Year'].min()}-{df['Year'].max()}",
            "Total Crimes Recorded": int(df["Total_Crimes"].sum()),
        }
        
        if "Crime_Solved" in df.columns:
            stats_data["Total Crimes Solved"] = int(df["Crime_Solved"].sum())
            stats_data["Overall Solved Rate %"] = round(
                (df["Crime_Solved"].sum() / df["Total_Crimes"].sum() * 100), 2
            )
        
        if "Pending_Percentage" in df.columns:
            stats_data["Avg Pending %"] = round(df["Pending_Percentage"].mean(), 2)
            stats_data["Max Pending %"] = round(df["Pending_Percentage"].max(), 2)
            stats_data["Min Pending %"] = round(df["Pending_Percentage"].min(), 2)

        # ----------------------------------------------------------
        # ANALYSIS 7: State-wise Summary Table
        # ----------------------------------------------------------
        if "Crime_Solved" in df.columns and "Pending_Percentage" in df.columns:
            summary_table = df.groupby("State").agg({
                "Total_Crimes": "sum",
                "Crime_Solved": "sum",
                "Pending_Percentage": "mean",
                "Crime_Rate": "mean"
            }).reset_index()
            
            summary_table["Solved_Rate %"] = (
                summary_table["Crime_Solved"] / summary_table["Total_Crimes"] * 100
            ).round(2)
            
            summary_table = summary_table.rename(columns={
                "Total_Crimes": "Total_Crimes",
                "Crime_Solved": "Crimes_Solved",
                "Pending_Percentage": "Avg_Pending_%",
                "Crime_Rate": "Avg_Crime_Rate"
            })
            
            summary_table = summary_table.sort_values("Solved_Rate %", ascending=False)
            table_html = summary_table.to_html(
                classes="table table-striped table-sm",
                index=False,
                float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x
            )
        else:
            table_html = None

        # ----------------------------------------------------------
        # SEND TO TEMPLATE
        # ----------------------------------------------------------
        context = {
            "stats": stats_data,
            "graphs": {
                "Crime Solved Rate by State": "/static/graphs/crime_solved_rate.png",
                "Top 15 States with Pending Cases": "/static/graphs/pending_cases.png",
                "Solved vs Pending Trend": "/static/graphs/solved_vs_pending_trend.png",
                "Correlation Matrix": "/static/graphs/correlation_matrix.png",
                "Top Performers": "/static/graphs/top_performers.png",
            },
            "summary_table": table_html,
        }

        return render(request, "advanced_analysis.html", context)

    except Exception as e:
        print("ADVANCED ANALYSIS ERROR:", e)
        traceback.print_exc()
        messages.error(request, f"Analysis failed: {str(e)}")
        return redirect("/upload/")
