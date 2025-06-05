# File: app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import plotly.express as px

# --- Caching for speed ---
@st.cache_data
def load_data():
    return pd.read_csv('blood_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

# --- Load Data and Model ---
df = load_data()
model = load_model()

# --- Data Preprocessing ---
df.fillna(0, inplace=True)
df.drop_duplicates(inplace=True)
df['Country'] = df['Country'].str.strip()
df.columns = df.columns.str.strip()  # Remove whitespace from column names

st.write("Available columns:", df.columns.tolist())

# Rename if necessary
if 'continent' in df.columns:
    df.rename(columns={'continent': 'Continent'}, inplace=True)
elif 'CONTINENT' in df.columns:
    df.rename(columns={'CONTINENT': 'Continent'}, inplace=True)

le = LabelEncoder()
df['Country_Num'] = le.fit_transform(df['Country'])

bloodtype_columns = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
df['Total_Rh_Pos'] = df[['O+', 'A+', 'B+', 'AB+']].sum(axis=1)
df['Total_Rh_Neg'] = df[['O-', 'A-', 'B-', 'AB-']].sum(axis=1)
df['Rarest_Blood_Type'] = df[bloodtype_columns].idxmin(axis=1)
df['Most_Common_Blood_Type'] = df[bloodtype_columns].idxmax(axis=1)

scaler = MinMaxScaler()
cols_to_normalize = ['Population'] + bloodtype_columns + ['Total_Rh_Pos', 'Total_Rh_Neg']
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# --- UI ---
st.sidebar.header("Blood Type Matching Tool")
blood_types = ['A+', 'O+', 'B+', 'AB+', 'A-', 'B-', 'O-', 'AB-']
selected_blood = st.sidebar.selectbox("Select Your Blood Type", blood_types)
role = st.sidebar.radio("Are you a...", ['Donor', 'Recipient'])

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

# --- Prediction Logic ---
def get_predictions(blood_type, role):
    results = {}
    for continent in continents:
        # Example input format - adjust to match what your model expects
        input_features = pd.DataFrame([{
            'Continent': continent,
            'Blood Type': blood_type,
            'Role': role
        }])
        # Assuming model.predict_proba is valid for this input
        try:
            prob = model.predict_proba(input_features)[0][1] * 100
        except:
            prob = 0
        results[continent] = round(prob, 2)
    return results

if st.sidebar.button("Submit"):
    # --- Display World Overview First ---
    st.header(f"🌍 Global Overview for {selected_blood} ({role})")

    world_data = df[df['Country'] == 'World'].copy()
    if not world_data.empty:
        # Bar Graph of Blood Type Distribution
        blood_counts = world_data[bloodtype_columns].T.reset_index()
        blood_counts.columns = ['Blood Type', 'Proportion']
        st.plotly_chart(px.bar(blood_counts, x='Blood Type', y='Proportion', title="World Blood Type Proportions"))

        # Most and Least Common
        most_common = world_data[bloodtype_columns].idxmax(axis=1).values[0]
        least_common = world_data[bloodtype_columns].idxmin(axis=1).values[0]
        st.success(f"Most Common Blood Type in the World: **{most_common}**")
        st.warning(f"Rarest Blood Type in the World: **{least_common}**")
    else:
        st.warning("No global data available.")

    # --- Continent Tabs ---
    tab_list = st.tabs(continents)
    predictions = get_predictions(selected_blood, role)

    for idx, continent in enumerate(continents):
        with tab_list[idx]:
            st.header(f"{continent} Overview for {selected_blood} ({role})")

            continent_data = df[df['Continent'] == continent].copy()
            continent_data['Blood Type'] = selected_blood
            continent_data['Count'] = continent_data[selected_blood]

            if continent_data.empty:
                st.warning(f"No data available for {continent}.")
                continue

            # Map Plot
            map_fig = px.scatter_geo(
                continent_data,
                locations="Country",
                locationmode="country names",
                hover_name="Country",
                color_discrete_sequence=['red'],
                scope="world",
                title=f"Countries in {continent}"
            )
            st.plotly_chart(map_fig)

            # Bar Chart - Top 5 Countries
            top_5 = continent_data.nlargest(5, 'Count')
            if top_5.empty:
                st.warning(f"No {selected_blood} blood type data for top countries in {continent}.")
            else:
                bar_fig = px.bar(
                    top_5,
                    x='Country',
                    y='Count',
                    color='Country',
                    title=f"Top 5 Countries with {selected_blood}"
                )
                st.plotly_chart(bar_fig)

            # Pie Chart
            pie_data = pd.melt(
                continent_data,
                id_vars=['Country'],
                value_vars=bloodtype_columns,
                var_name='Blood Type',
                value_name='Count'
            )
            pie_fig = px.pie(
                pie_data,
                names='Blood Type',
                values='Count',
                title=f"Blood Type Distribution in {continent}"
            )
            st.plotly_chart(pie_fig)

            # Prediction Output
            if role == 'Donor':
                st.success(f"Your blood type is needed by approximately **{predictions[continent]}%** of the population in {continent}.")
            else:
                st.info(f"You have a **{predictions[continent]}%** chance of finding a donor match in {continent}.")
