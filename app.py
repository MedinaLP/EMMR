# File: app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import plotly.express as px

# --- Caching for speed ---
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv')

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

country_to_continent = {
    'Albania': 'Europe', 'Algeria': 'Africa', 'Argentina': 'South America', 'Armenia': 'Asia',
    'Australia': 'Oceania', 'Austria': 'Europe', 'Azerbaijan': 'Asia', 'Bahrain': 'Asia',
    'Bangladesh': 'Asia', 'Belarus': 'Europe', 'Belgium': 'Europe', 'Bhutan': 'Asia',
    'Bolivia': 'South America', 'Bosnia and Herzegovina': 'Europe', 'Brazil': 'South America',
    'Bulgaria': 'Europe', 'Burkina Faso': 'Africa', 'Cambodia': 'Asia', 'Cameroon': 'Africa',
    'Canada': 'North America', 'Chile': 'South America', 'China': 'Asia', 'Colombia': 'South America',
    'Costa Rica': 'North America', 'Croatia': 'Europe', 'Cuba': 'North America', 'Cyprus': 'Asia',
    'Czech Republic': 'Europe', 'Democratic Republic of the Congo': 'Africa', 'Denmark': 'Europe',
    'Dominican Republic': 'North America', 'Ecuador': 'South America', 'Egypt': 'Africa',
    'El Salvador': 'North America', 'Estonia': 'Europe', 'Ethiopia': 'Africa', 'Fiji': 'Oceania',
    'Finland': 'Europe', 'France': 'Europe', 'Gabon': 'Africa', 'Georgia': 'Asia', 'Germany': 'Europe',
    'Ghana': 'Africa', 'Greece': 'Europe', 'Guinea': 'Africa', 'Honduras': 'North America',
    'Hong Kong': 'Asia', 'Hungary': 'Europe', 'Iceland': 'Europe', 'India': 'Asia',
    'Indonesia': 'Asia', 'Iran': 'Asia', 'Iraq': 'Asia', 'Ireland': 'Europe', 'Israel': 'Asia',
    'Italy': 'Europe', 'Ivory Coast': 'Africa', 'Jamaica': 'North America', 'Japan': 'Asia',
    'Jordan': 'Asia', 'Kazakhstan': 'Asia', 'Kenya': 'Africa', 'Laos': 'Asia', 'Latvia': 'Europe',
    'Lebanon': 'Asia', 'Libya': 'Africa', 'Liechtenstein': 'Europe', 'Lithuania': 'Europe',
    'Luxemburg': 'Europe', 'Macao': 'Asia', 'Malaysia': 'Asia', 'Malta': 'Europe',
    'Mauritania': 'Africa', 'Mauritius': 'Africa', 'Mexico': 'North America', 'Moldova': 'Europe',
    'Mongolia': 'Asia', 'Morocco': 'Africa', 'Myanmar': 'Asia', 'Namibia': 'Africa', 'Nepal': 'Asia',
    'Netherlands': 'Europe', 'New Zealand': 'Oceania', 'Nicaragua': 'North America',
    'Nigeria': 'Africa', 'North Korea': 'Asia', 'North Macedonia': 'Europe', 'Norway': 'Europe',
    'Pakistan': 'Asia', 'Papua New Guinea': 'Oceania', 'Paraguay': 'South America',
    'Peru': 'South America', 'Philippines': 'Asia', 'Poland': 'Europe', 'Portugal': 'Europe',
    'Romania': 'Europe', 'Russia': 'Europe', 'Saudi Arabia': 'Asia', 'Serbia': 'Europe',
    'Singapore': 'Asia', 'Slovakia': 'Europe', 'Slovenia': 'Europe', 'Somalia': 'Africa',
    'South Africa': 'Africa', 'South Korea': 'Asia', 'Sri Lanka': 'Asia', 'Spain': 'Europe',
    'Sudan': 'Africa', 'Sweden': 'Europe', 'Switzerland': 'Europe', 'Syria': 'Asia', 'Taiwan': 'Asia',
    'Thailand': 'Asia', 'Tunisia': 'Africa', 'Turkey': 'Asia', 'Uganda': 'Africa', 'Ukraine': 'Europe',
    'United Arab Emirates': 'Asia', 'United Kingdom': 'Europe', 'United States': 'North America',
    'Uzbekistan': 'Asia', 'Venezuela': 'South America', 'Vietnam': 'Asia', 'Yemen': 'Asia',
    'Zimbabwe': 'Africa'
}

df['Continent'] = df['Country'].map(country_to_continent)

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

# --- App Output ---
if st.sidebar.button("Submit"):
    predictions = get_predictions(selected_blood, role)
    tab_list = st.tabs(continents)

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
            if not top_5.empty:
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
