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


bloodtype_columns = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']

# --- UI ---
st.sidebar.header("Blood Type Probability Tool")
blood_types = ['A+', 'O+', 'B+', 'AB+', 'A-', 'B-', 'O-', 'AB-']
selected_blood = st.sidebar.selectbox("Select Your Blood Type", blood_types)
role = st.sidebar.radio("Are you a...", ['Donor', 'Recipient'])

# Predictions
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
    predictions = get_predictions(selected_blood, role)
    tab_list = st.tabs(continents)

    for idx, continent in enumerate(continents):
        with tab_list[idx]:
            st.header(f"{continent}") 
            st.subheader(Overview for {selected_blood} ({role})")
            continent_data = df[df['Continent'] == continent].copy()
            continent_data['Blood Type'] = selected_blood
            continent_data['Count'] = continent_data[selected_blood]

            if continent_data.empty:
                st.warning(f"No data available for {continent}.")
                continue

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
