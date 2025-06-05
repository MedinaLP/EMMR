# File: app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- Load the model and data from current notebook environment ---
try:
    model = xgb_model  # The trained model must be in memory
    df = blood_data     # The cleaned dataset must be assigned to 'df'
except NameError:
    st.error("Make sure to define 'xgb_model' and 'blood_data' (as df) in the notebook environment before running the app.")
    st.stop()

# Data Cleaning
df.fillna(0, inplace=True)
df.drop_duplicates(inplace=True)
df['Country'] = df['Country'].str.strip()

# Encoding country names
le = LabelEncoder()
df['Country_Num'] = le.fit_transform(df['Country'])

# Feature Engineering
bloodtype_columns = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
df['Total_Rh_Pos'] = df[['O+', 'A+', 'B+', 'AB+']].sum(axis=1)
df['Total_Rh_Neg'] = df[['O-', 'A-', 'B-', 'AB-']].sum(axis=1)
df['Rarest_Blood_Type'] = df[bloodtype_columns].idxmin(axis=1)
df['Most_Common_Blood_Type'] = df[bloodtype_columns].idxmax(axis=1)

# Normalization
scaler = MinMaxScaler()
cols_to_normalize = ['Population'] + bloodtype_columns + ['Total_Rh_Pos', 'Total_Rh_Neg']
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Sidebar Inputs
continent_coords = {
    'Africa': (1.6508, 10.2679),
    'Asia': (34.0479, 100.6197),
    'Europe': (54.5260, 15.2551),
    'North America': (54.5260, -105.2551),
    'South America': (-8.7832, -55.4915),
    'Oceania': (-25.2744, 133.7751),
    'Antarctica': (-82.8628, 135.0000)
}
continents = list(continent_coords.keys())
blood_types = ['A+', 'O+', 'B+', 'AB+', 'A-', 'B-', 'O-', 'AB-']

st.sidebar.header("Blood Type Matching Tool")
selected_blood = st.sidebar.selectbox("Select Your Blood Type", blood_types)
role = st.sidebar.radio("Are you a...", ['Donor', 'Recipient'])

# Prediction Function
def get_predictions(blood_type, role):
    results = {}
    for continent in continents:
        input_df = pd.DataFrame({
            'Blood Type': [blood_type],
            'Role': [role],
            'Continent': [continent]
        })
        pred = model.predict_proba(input_df)[0][1] * 100
        results[continent] = round(pred, 2)
    return results

# Submit Button
if st.sidebar.button("Submit"):
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
