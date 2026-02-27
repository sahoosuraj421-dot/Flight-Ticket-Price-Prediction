import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from joblib import load

# ---------------------- Page Setup ---------------------- #
st.set_page_config(
    page_title="Flight Ticket Price Prediction",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------- Custom CSS Styling ---------------------- #
custom_css = """
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    [data-testid="stMain"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Typography */
    h1 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        color: #1a1a2e !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
        letter-spacing: -0.5px !important;
    }
    
    h2 {
        color: #16213e !important;
        font-weight: 600 !important;
        border-bottom: 3px solid #667eea !important;
        padding-bottom: 0.5rem !important;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        color: #2d3561 !important;
        font-weight: 600 !important;
        margin-top: 1.2rem !important;
    }
    
    p {
        color: #445566 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    
    /* Container styling */
    [data-testid="column"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="column"]:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    /* Input fields */
    input, select, textarea {
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, select:focus, textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Alert/Info boxes */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border-left: 5px solid !important;
        padding: 1rem 1.2rem !important;
        font-size: 15px !important;
    }
    
    /* Success alert */
    [data-testid="stAlert"] > div > div > div > p {
        font-weight: 500 !important;
        color: #155724 !important;
    }
    
    [data-testid="stAlert"] {
        background-color: #d4edda !important;
        border-left-color: #28a745 !important;
    }
    
    /* Info alert */
    [data-testid="stAlert"] {
        background-color: #d1ecf1 !important;
        border-left-color: #17a2b8 !important;
    }
    
    /* Warning alert */
    [data-testid="stAlert"] {
        background-color: #fff3cd !important;
        border-left-color: #ffc107 !important;
    }
    
    /* Error alert */
    [data-testid="stAlert"] {
        background-color: #f8d7da !important;
        border-left-color: #dc3545 !important;
    }
    
    /* Buttons */
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        cursor: pointer !important;
    }
    
    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    [data-testid="stButton"] > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Divider */
    hr {
        border: none !important;
        border-top: 2px solid rgba(102, 126, 234, 0.3) !important;
        margin: 2rem 0 !important;
    }
    
    /* Labels */
    label {
        font-weight: 600 !important;
        color: #2d3561 !important;
        font-size: 14px !important;
    }
    
    /* Selectbox and input wrappers */
    [data-testid="stSelectbox"], 
    [data-testid="stDateInput"],
    [data-testid="stNumberInput"] {
        padding: 5px !important;
    }
    
    /* Text styling for headers */
    [data-testid="stMarkdownContainer"] h1 {
        margin-bottom: 0.5rem !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.8);
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Main header with enhanced styling
st.markdown(
    "<h1 style='text-align:center; white-space: nowrap; font-size: 3.5rem; margin-bottom: 0.5rem;'>✈️ Flight Ticket Price Prediction 🎟️</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size: 18px; color: #445566; margin-bottom: 2rem;'>💡 Smart ML-based airfare estimation system</p>",
    unsafe_allow_html=True
)

# ---------------------- Load Artifacts ---------------------- #
@st.cache_resource
def load_artifacts():
    model = load("flight_price_model.pkl")
    categorical_columns = load("categorical_features.pkl")
    numerical_columns = load("numeric_features.pkl")
    return model, categorical_columns, numerical_columns

@st.cache_data
def load_route_summary():
    df = pd.read_csv("route_summary.csv")
    df.columns = df.columns.str.upper()  # Ensure uppercase columns
    return df

model, categorical_columns, numerical_columns = load_artifacts()
route_summary = load_route_summary()

input_data = {}

# ---------------------- Helpers ---------------------- #
def hours_to_hrs_mins(hours_float):
    hrs = int(hours_float)
    mins = int(round((hours_float - hrs) * 60))
    if mins == 60:
        hrs += 1
        mins = 0
    return hrs, mins

def get_time_bucket(hour):
    if 5 <= hour < 9:
        return "Early Morning"
    elif 9 <= hour < 13:
        return "Morning"
    elif 13 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    elif 21 <= hour < 24:
        return "Night"
    else:
        return "Late Night"

# ---------------------- Route Selection ---------------------- #
st.markdown("### 🔍 Route")

c1, c2 = st.columns(2)

with c1:
    input_data['SOURCE CITY'] = st.selectbox(
        "From",
        sorted(route_summary['SOURCE CITY'].unique())
    )

with c2:
    input_data['DESTINATION CITY'] = st.selectbox(
        "To",
        sorted(route_summary['DESTINATION CITY'].unique())
    )

# Get route info
route = route_summary[
    (route_summary['SOURCE CITY'] == input_data['SOURCE CITY']) &
    (route_summary['DESTINATION CITY'] == input_data['DESTINATION CITY'])
]

if not route.empty:
    duration = float(route['MIN_DURATION'].iloc[0])
else:
    st.warning("⚠️ Route not found. Using default duration of 1 hr.")
    duration = 1.0

input_data['DURATION'] = duration
hrs, mins = hours_to_hrs_mins(duration)

# Display route with duration below it
st.markdown(
    f"<h3 style='text-align:center;'>"
    f"{input_data['SOURCE CITY']} ✈️ {input_data['DESTINATION CITY']}"
    f"</h3>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='text-align:center; font-size:16px;'>Duration: {hrs} hrs {mins} mins</p>",
    unsafe_allow_html=True
)

# ---------------------- Dates ---------------------- #
st.markdown("### 📅 Travel Dates")

d1, d2 = st.columns(2)

with d1:
    booking_date = st.date_input("Booking Date", value=datetime.today())

with d2:
    flight_date = st.date_input(
        "Flight Date",
        value=booking_date + timedelta(days=1),
        min_value=booking_date + timedelta(days=1)
    )

days_left = (flight_date - booking_date).days

input_data['BOOKING YEAR'] = booking_date.year
input_data['BOOKING MONTH'] = booking_date.month
input_data['BOOKING DAY'] = booking_date.day
input_data['FLIGHT YEAR'] = flight_date.year
input_data['FLIGHT MONTH'] = flight_date.month
input_data['FLIGHT DAY'] = flight_date.day
input_data['DAYS LEFT'] = days_left

st.info(f"🕒 Days until departure: **{days_left} days**")

# ---------------------- Flight Details ---------------------- #
st.markdown("### 🧳 Flight Details")

f1, f2, f3 = st.columns(3)

with f1:
    input_data['AIRLINE'] = st.selectbox(
        "Airline",
        ['SpiceJet', 'AirAsia', 'Vistara', 'GO FIRST', 'Indigo', 'Air India']
    )

with f2:
    input_data['STOPS'] = st.selectbox(
        "Stops",
        ['Zero', 'One', 'Two or More']
    )

with f3:
    input_data['PRICE CLASS'] = st.selectbox(
        "Class",
        ['Economy', 'Premium Economy', 'Business', 'First Class', 'Luxury']
    )

# ---------------------- Time Logic ---------------------- #
st.markdown("### ⏰ Time")

departure_time = st.selectbox(
    "Departure Time",
    ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night']
)
input_data['DEPARTURE TIME'] = departure_time

departure_hour_map = {
    'Early Morning': 6,
    'Morning': 10,
    'Afternoon': 14,
    'Evening': 18,
    'Night': 22,
    'Late Night': 23
}

dep_hour = departure_hour_map[departure_time]

arrival_dt = datetime(2000, 1, 1, dep_hour, 0) + timedelta(hours=duration)
arrival_bucket = get_time_bucket(arrival_dt.hour)

input_data['ARRIVAL TIME'] = arrival_bucket

st.text_input(
    "Arrival Time (Auto)",
    value=f"{arrival_bucket} ({arrival_dt.strftime('%H:%M')})",
    disabled=True
)

# ---------------------- Passengers ---------------------- #
st.markdown("### 👥 Passengers")

passengers = st.number_input("Number of Passengers", 1, 9, 1)

# ---------------------- Prediction ---------------------- #
st.markdown("---")

if st.button("🔮 Predict Ticket Price", use_container_width=True):

    X_columns = [
        'AIRLINE', 'SOURCE CITY', 'STOPS', 'DESTINATION CITY', 'PRICE CLASS',
        'BOOKING YEAR', 'BOOKING MONTH', 'BOOKING DAY', 'DAYS LEFT',
        'FLIGHT YEAR', 'FLIGHT MONTH', 'FLIGHT DAY',
        'DEPARTURE TIME', 'DURATION', 'ARRIVAL TIME'
    ]

    input_df = pd.DataFrame([[input_data[col] for col in X_columns]], columns=X_columns)

    try:
        base_price = model.predict(input_df)[0]

        cgst = base_price * 0.025
        sgst = base_price * 0.025
        ticket_price = base_price + cgst + sgst
        total_price = ticket_price * passengers

        st.success(f"🎫 Price per Ticket: ₹ {ticket_price:,.2f}")
        st.info(f"🧾 CGST (2.5%): ₹ {cgst:,.2f} | SGST (2.5%): ₹ {sgst:,.2f}")
        st.success(f"💰 Total Price ({passengers} passengers): ₹ {total_price:,.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")