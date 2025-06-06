# File: app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import plotly.express as px
from xgboost import XGBClassifier

# --- Caching for speed ---
@st.cache_data
def load_data():
    return pd.read_csv('blood_data.csv')

@st.cache_data
def load_world_data():
    return pd.read_csv('world_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

# --- Load Data and Model ---
df = load_data()
world_data = load_world_data()
model = load_model()

le = LabelEncoder()
df['Country_Num'] = le.fit_transform(df['Country'])
bloodtype_columns = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']

# --- Data Preprocessing ---
df.fillna(0, inplace=True)
df.drop_duplicates(inplace=True)
df['Country'] = df['Country'].str.strip()
df.columns = df.columns.str.strip()  # Remove whitespace from column names

# Define continent coordinates (optional for maps)
continent_coords = {
    'Africa': (1.6508, 10.2679),
    'Asia': (34.0479, 100.6197),
    'Europe': (54.5260, 15.2551),
    'North America': (54.5260, -105.2551),
    'South America': (-8.7832, -55.4915),
    'Oceania': (-25.2744, 133.7751)
}
continents = list(continent_coords.keys())

# --- UI ---
st.sidebar.header("Blood Type Probability Tool")
blood_types = ['A+', 'O+', 'B+', 'AB+', 'A-', 'B-', 'O-', 'AB-']
selected_blood = st.sidebar.selectbox("Select Your Blood Type", blood_types)
role = st.sidebar.radio("Are you a...", ['Donor', 'Recipient'])
submitted = st.sidebar.button("Submit")

if submitted:
    st.header(f"🩸Showing Compatibility Map for {selected_blood} ({role})")

    tab_list = st.tabs(continents)
    #predictions = get_predictions(selected_blood, role)

    for idx, continent in enumerate(continents):
        with tab_list[idx]:
            st.header(f"{continent}")
            st.subheader(f"Overview for {selected_blood} ({role})")

           # Filter data for this continent
            continent_data = df[df['Continent'] == continent].copy()

            if continent_data.empty:
                st.warning(f"No data available for {continent}.")
                continue

            # ==========================
            # MAP: show prevalence of selected blood type
            # ==========================
            map_fig = px.choropleth(
                continent_data,
                locations="Country",
                locationmode="country names",
                color=selected_blood,
                hover_name="Country",
                color_continuous_scale = "Reds" if role.lower() == "donor" else "Blues",
                title=f"{selected_blood} Blood Type Distribution in {continent}",
                scope="world"
            )
            st.plotly_chart(map_fig, use_container_width=True)

            # ==========================
            # BAR: top / bottom 5 countries
            # ==========================
            if role == "Donor":
                bar_data = continent_data.nsmallest(5, selected_blood)
                bar_title = f"Top 5 Countries needing {selected_blood} blood type"
            else:
                bar_data = continent_data.nlargest(5, selected_blood)
                bar_title = f"Top 5 Countries with highest {selected_blood} blood type"

            bar_fig = px.bar(
                bar_data,
                x="Country",
                y=selected_blood,
                color="Country",
                title=bar_title,
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # ==========================
            # PIE: blood type distribution (mean across countries)
            # ==========================
            pie_vals = continent_data[bloodtype_columns].mean().reset_index()
            pie_vals.columns = ["Blood Type", "Mean Proportion"]
            pie_fig = px.pie(
                pie_vals,
                names="Blood Type",
                values="Mean Proportion",
                title=f"Average Blood Type Distribution in {continent}",
            )
            st.plotly_chart(pie_fig, use_container_width=True)
            
            # ==========================
            # TEXT PREDICTION (simple heuristic based on prevalence)
            # ==========================
            selected_row = pie_vals[pie_vals['Blood Type'] == selected_blood]
            if not selected_row.empty:
                pie_pct = selected_row['Mean Proportion'].values[0] * 100  # Convert to percentage
            
                if role.lower() == "donor":
                    demand_pct = round(100 - pie_pct, 2)
                    st.success(
                        f"Your blood type is needed by approximately **{demand_pct}%** of the population in {continent}."
                    )
                else:
                    avail_pct = round(pie_pct, 2)
                    st.info(
                        f"You have a **{avail_pct}%** chance of finding a donor match in {continent}."
                    )
            else:
                st.warning("Selected blood type data not found in pie chart distribution.")
    
else:
     # --- Starting Page with Global Overview ---
    st.title("🌍 Data-driven Global Prediction of Blood Type Probabilities for Donors and Patients")
    
    if not world_data.empty:
        blood_counts = world_data[bloodtype_columns].T.reset_index()
        blood_counts.columns = ['Blood Type', 'Proportion']
    
        st.subheader("World Blood Type Distribution")
        fig = px.bar(
            blood_counts,
            x='Blood Type',
            y='Proportion',
            color='Blood Type',
            title="Distribution of Blood Types Globally",
            labels={'Proportion': 'Proportion (Normalized)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
        most_common = blood_counts.loc[blood_counts['Proportion'].idxmax(), 'Blood Type']
        least_common = blood_counts.loc[blood_counts['Proportion'].idxmin(), 'Blood Type']
        st.success(f"Most Common Blood Type in the World: **{most_common}**")
        st.warning(f"Rarest Blood Type in the World: **{least_common}**")
    else:
        st.warning("No global data available.")
