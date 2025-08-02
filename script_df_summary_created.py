import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px
import re
import plotly.graph_objects as go
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_executive_summary(data_description):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are a healthcare strategy assistant. Based on the following data insights, generate a concise executive summary for hospital executives. Highlight key patterns, outliers, competitive insights, and strategic recommendations.

    Data Summary:
    {data_description}
    
    Format:
    1. Key Trends:
    2. Competitive Positioning:
    3. Strategic Risks/Opportunities:
    4. Suggested Actions:
    """

    response = model.generate_content(prompt)
    return response.text

@st.cache_data
def load_excel_data():
    url1 = "https://data.chhs.ca.gov/dataset/0fa47cc2-284e-45a2-83b6-a4524fd84fb2/resource/a6ed2963-df4a-4668-9ead-6d79310a7686/download/2023top25msdrg_pivot_202407018.xlsm"
    url2 = "https://data.chhs.ca.gov/dataset/0fa47cc2-284e-45a2-83b6-a4524fd84fb2/resource/91f93719-307c-464f-9852-5b892525fea2/download/2022top25msdrg_pivot_20230701.xlsm"
    url3 = "https://data.chhs.ca.gov/dataset/0fa47cc2-284e-45a2-83b6-a4524fd84fb2/resource/e0e070f3-79e8-43d5-9d3d-5880191208fe/download/2021top25msdrg_pivot_20220705.xlsm"
    url4 = "https://gis.dhcs.ca.gov/api/download/v1/items/a83122d79b1f4b85b3637d8a462fc2d9/csv?layers=7"

    url_payer= "https://data.chhs.ca.gov/dataset/44f54d54-e12c-4c06-b32e-924ecf4410d5/resource/6365cd77-61c5-4cda-a6cb-4f052fcf3949/download/expectedpayerpdd.csv"

    headers = {"User-Agent": "Mozilla/5.0"}

    # Load Excel datasets
    response1 = requests.get(url1, headers=headers)
    df1 = pd.read_excel(io.BytesIO(response1.content), sheet_name="All Data")
    df1["Year"] = "2023"

    response2 = requests.get(url2, headers=headers)
    df2 = pd.read_excel(io.BytesIO(response2.content), sheet_name="All Data")
    df2["Year"] = "2022"

    response3 = requests.get(url3, headers=headers)
    df3 = pd.read_excel(io.BytesIO(response3.content), sheet_name="All Data")
    df3["Year"] = "2021"

    response4 = requests.get(url4, headers=headers)
    df4 = pd.read_csv(io.StringIO(response4.text))

    df4["DHCS_County_Code"] = df4["DHCS_County_Code"].astype(int)

    response_payer = requests.get(url_payer, headers=headers)
    df_payer = pd.read_csv(io.StringIO(response_payer.text))

    df5 = pd.read_csv("Mapped_DRG_Service_Lines.csv")

    # Example: load your data
    df_jobs = pd.read_csv("kaiser_sutter_jobs.csv")

    
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    combined_df["Year"] = combined_df["Year"].astype(int)
    combined_df = combined_df.merge(df5,on="DRGDescription",how="left")
    
    combined_df["COUNTY_CODE"] = pd.to_numeric(combined_df["COUNTY_CODE"], errors='coerce').astype("Int64")
    combined_df = combined_df.merge(
    df4[["DHCS_County_Code", "County_Name"]],
    left_on="COUNTY_CODE",
    right_on="DHCS_County_Code",
    how="left"
)
    return combined_df, df_payer, df_jobs

df, df_payer, df_jobs = load_excel_data()

df = df[df["FacilityName"] != "00_Statewide"]

df_summary = df.groupby(
    ["Year", "FacilityName", "DRGDescription", "County_Name", "ServiceLine"],
    as_index=False
)[["Discharges", "ValidCharges"]].sum()

dx_options = sorted(df_summary["DRGDescription"].unique())
sl_options = ["All"] + sorted(df_summary["ServiceLine"].dropna().unique())
facility_options = sorted(df_summary["FacilityName"].unique())
county_options = sorted(df_summary["County_Name"].unique())

st.title("📈 Service Line Performance Benchmarking")

selected_facilities = st.multiselect(
    "Select Facility",
    options=facility_options,
    default=["KAISER FOUNDATION HOSPITAL - SOUTH SACRAMENTO", "SUTTER MEDICAL CENTER, SACRAMENTO", "UNIVERSITY OF CALIFORNIA DAVIS MEDICAL CENTER"]
)

selected_sl = st.multiselect(
    "Select Service Line",
    options=sl_options,
    default=["Maternity & Newborn"]
)

from plotly.subplots import make_subplots

# Filter based on selections
filtered_df = df_summary.copy()
if selected_facilities:
    filtered_df = filtered_df[filtered_df["FacilityName"].isin(selected_facilities)]
if selected_sl and "All" not in selected_sl:
    filtered_df = filtered_df[filtered_df["ServiceLine"].isin(selected_sl)]

# Group data by year and sum values
# yearly_summary = filtered_df.groupby("Year", as_index=False)[["Discharges", "ValidCharges"]].sum()
# yearly_summary["Year"] = yearly_summary["Year"].astype(str)  # Treat year as categorical

yearly_summary = (
    filtered_df.groupby(["Year", "FacilityName"], as_index=False)[["Discharges", "ValidCharges"]]
    .sum()
)
yearly_summary["Year"] = yearly_summary["Year"].astype(str)



# If only one facility is selected, keep the name
if len(selected_facilities) == 1:
    yearly_summary["Facility"] = selected_facilities[0]
else:
    yearly_summary["Facility"] = "Multiple"

# Create dual-axis plot
# fig = make_subplots(specs=[[{"secondary_y": True}]])

# fig.add_trace(
#     go.Bar(
#         x=yearly_summary["Year"],
#         y=yearly_summary["Discharges"],
#         name="Discharges",
#         hovertemplate=
#             "Year: %{x}<br>" +
#             "Facility: %{customdata[0]}<br>" +
#             "Discharges: %{y}<extra></extra>",
#         customdata=yearly_summary[["Facility"]]
#     ),
#     secondary_y=False,
# )

# fig.add_trace(
#     go.Scatter(
#         x=yearly_summary["Year"],
#         y=yearly_summary["ValidCharges"],
#         name="Valid Charges",
#         mode='lines+markers',
#         hovertemplate=
#             "Year: %{x}<br>" +
#             "Facility: %{customdata[0]}<br>" +
#             "Valid Charges: $%{y:,.0f}<extra></extra>",
#         customdata=yearly_summary[["Facility"]]
#     ),
#     secondary_y=True,
# )

show_forecast = st.checkbox("Show Forecast (Next 2 Years)", value=False)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_series(series, steps=2):
    # Fit ETS model (no trend/seasonal by default)
    model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
    fit = model.fit()
    forecast = fit.forecast(steps)
    return forecast

fig = make_subplots(specs=[[{"secondary_y": True}]])

for facility in selected_facilities:
    df_fac = yearly_summary[yearly_summary["FacilityName"] == facility].copy()
    df_fac = df_fac.sort_values("Year")  # Ensure chronological

    # Plot actual Discharges
    fig.add_trace(
        go.Bar(
            x=df_fac["Year"],
            y=df_fac["Discharges"],
            name=f"{facility} - Discharges",
            hovertemplate=(
    rf"Year: %{{x}}<br>Facility: {facility}<br>Discharges (Forecast): %{{y}}<extra></extra>"
)

        ),
        secondary_y=False
    )

    # Plot actual Valid Charges
    fig.add_trace(
        go.Scatter(
            x=df_fac["Year"],
            y=df_fac["ValidCharges"],
            name=f"{facility} - Valid Charges",
            mode="lines+markers",
            hovertemplate=(
    rf"Year: %{{x}}<br>Facility: {facility}<br>Valid Charges (Forecast): $%{{y:,.0f}}<extra></extra>"
)

        ),
        secondary_y=True
    )

    # ✅ Forecast section (only if checkbox is enabled)
    if show_forecast and len(df_fac) >= 3:
        last_year = int(df_fac["Year"].max())
        future_years = [str(last_year + i) for i in range(1, 3)]

        # Forecast Discharges
        discharge_fc = forecast_series(df_fac.set_index("Year")["Discharges"])
        fig.add_trace(
    go.Bar(
        x=future_years,
        y=discharge_fc.values,
        name=f"{facility} - Discharges Forecast",
        marker=dict(color="rgba(0, 0, 0, 0.4)"),  # semi-transparent gray
        hovertemplate=(
            rf"Year: %{{x}}<br>Facility: {facility}<br>Discharges (Forecast): %{{y}}<extra></extra>"
        )
    ),
    secondary_y=False
)

        # Forecast Valid Charges
        charges_fc = forecast_series(df_fac.set_index("Year")["ValidCharges"])
        fig.add_trace(
            go.Scatter(
                x=future_years,
                y=charges_fc.values,
                name=f"{facility} - Valid Charges Forecast",
                mode="lines+markers",
                line=dict(dash="dot"),
                hovertemplate=(
    rf"Year: %{{x}}<br>Facility: {facility}<br>Valid Charges (Forecast): $%{{y:,.0f}}<extra></extra>"
)

            ),
            secondary_y=True
        )

# fig.update_layout(
#     title="Yearly Discharges and Valid Charges",
#     xaxis_title="Year",
#     yaxis_title="Discharges",
#     legend_title="Metrics",
#     xaxis=dict(type='category')  # 👈 This forces categorical x-axis
# )
fig.update_layout(
    title="Yearly Discharges and Valid Charges by Facility",
    xaxis_title="Year",
    yaxis_title="Discharges",
    legend_title="Metrics",
    xaxis=dict(type='category'),
    legend=dict(
        orientation="h",       # horizontal legend
        yanchor="top",
        y=-0.3,                # push below plot
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=40, r=40, t=80, b=200),  # b=200 gives room for large legend
    height=700                            # taller chart to avoid crowding
)


fig.update_yaxes(title_text="Discharges", secondary_y=False)
fig.update_yaxes(title_text="Valid Charges ($)", secondary_y=True)
