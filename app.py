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
st.sidebar.header("Blood Type Probability Tool")
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

# --- Starting Page with Global Overview ---
st.title("🌍 Data-driven Global Prediction of Blood Type Probabilities for Donors and Patients")

# Show the bar graph only if World data is available
world_row = df[df['Country'].str.lower() == 'world']
if not world_row.empty:
    world_blood_data = world_row[bloodtype_columns].T.reset_index()
    world_blood_data.columns = ['Blood Type', 'Proportion']

    st.subheader("World Blood Type Distribution")
    fig = px.bar(
        world_blood_data,
        x='Blood Type',
        y='Proportion',
        color='Blood Type',
        title="Distribution of Blood Types in the World",
        labels={'Proportion': 'Proportion (Normalized)'}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No 'World' data row found in dataset.")
