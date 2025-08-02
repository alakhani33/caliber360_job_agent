import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px
import re
import plotly.graph_objects as go


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
    df_jobs = pd.read_csv("kaiser_jobs_revalidated.csv")

    
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

st.plotly_chart(fig, use_container_width=True)

#########################

# Year options
year_options = sorted(df_summary["Year"].unique(), reverse=True)
selected_year = st.selectbox(
    "Select Year",
    options=["All Years"] + [str(y) for y in year_options],
    index=1
)
if selected_year != "All Years":
    selected_year = int(selected_year)

# Sort selector
sort_field = st.selectbox(
    "Sort Market Share by",
    options=["Discharges", "ValidCharges"],
    index=0
)

# County selector
# selected_county = st.selectbox(
#     "Select County",
#     options=["All Counties"] + county_options,
#     index=0
# )
all_counties = ["All Counties"] + county_options
default_index = all_counties.index("Sacramento") if "Sacramento" in all_counties else 0

selected_county = st.selectbox(
    "Select County",
    options=all_counties,
    index=default_index
)


# Filter
market_df = df_summary.copy()

if selected_sl and "All" not in selected_sl:
    market_df = market_df[market_df["ServiceLine"].isin(selected_sl)]

if selected_county != "All Counties":
    market_df = market_df[market_df["County_Name"] == selected_county]

if selected_year != "All Years":
    market_df = market_df[market_df["Year"] == selected_year]

# Group, sort, top 10
market_summary = (
    market_df.groupby("FacilityName", as_index=False)[["Discharges", "ValidCharges"]]
    .sum()
    .sort_values(by=sort_field, ascending=False)
    .head(10)
)

# Plot
fig_market = px.bar(
    market_summary,
    x=sort_field,
    y="FacilityName",
    orientation="h",
    title=f"Top 10 Facilities by {sort_field}"
          f"{' | ' + str(selected_year) if selected_year != 'All Years' else ''}"
          f"{' | ' + selected_county if selected_county != 'All Counties' else ''}"
          f"{' | ' + ', '.join(selected_sl) if selected_sl and 'All' not in selected_sl else ''}",
    labels={"FacilityName": "Facility", sort_field: sort_field},
    height=600
)

fig_market.update_layout(yaxis={'categoryorder': 'total ascending'})

st.plotly_chart(fig_market, use_container_width=True)

################################

st.subheader("📉 Average Length of Stay Over Time")

alos_facilities = st.multiselect(
    "Select Facility for ALOS Chart",
    options=facility_options,
    default=["KAISER FOUNDATION HOSPITAL - SOUTH SACRAMENTO", "SUTTER MEDICAL CENTER, SACRAMENTO", "UNIVERSITY OF CALIFORNIA DAVIS MEDICAL CENTER"]
)

alos_service_lines = st.multiselect(
    "Select Service Line for ALOS Chart",
    options=sl_options,
    default=["Maternity & Newborn"]
)

# Filter base df
alos_df = df[df["FacilityName"].isin(alos_facilities)]

if alos_service_lines and "All" not in alos_service_lines:
    alos_df = alos_df[alos_df["ServiceLine"].isin(alos_service_lines)]

# Group by Year and Facility, taking mean ALOS
alos_yearly = alos_df.groupby(["Year", "FacilityName"], as_index=False)["ALOS"].mean()
alos_yearly["Year"] = alos_yearly["Year"].astype(str)

fig_alos = go.Figure()

for facility in alos_facilities:
    data = alos_yearly[alos_yearly["FacilityName"] == facility]
    fig_alos.add_trace(
        go.Scatter(
            x=data["Year"],
            y=data["ALOS"],
            mode="lines+markers",
            name=facility,
            hovertemplate=(
                rf"Year: %{{x}}<br>Facility: {facility}<br>Avg LOS: %{{y:.2f}} days<extra></extra>"
            )
        )
    )

fig_alos.update_layout(
    title="Average Length of Stay by Year",
    xaxis_title="Year",
    yaxis_title="Avg Length of Stay (days)",
    xaxis=dict(type='category'),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.25,
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=40, r=40, t=80, b=160),
    height=500
)

st.plotly_chart(fig_alos, use_container_width=True)


######################

st.subheader("💹 DRG Valid Charges Over Time")

default_drgs = [
    drg for drg in dx_options
    if "CESAREAN SECTION" in drg.upper() or "VAGINAL DELIVERY" in drg.upper()
]

selected_drg_facilities = st.multiselect(
    "Select Facility for DRG Valid Charges Chart",
    options=facility_options,
    default=["SUTTER MEDICAL CENTER, SACRAMENTO", "UNIVERSITY OF CALIFORNIA DAVIS MEDICAL CENTER"]
)

selected_drgs = st.multiselect(
    "Select DRGs",
    options=dx_options,
    default=default_drgs
)

# Filter original df (not summary)
drg_df = df[df["FacilityName"].isin(selected_drg_facilities)]

if selected_drgs:
    drg_df = drg_df[drg_df["DRGDescription"].isin(selected_drgs)]

# Total ValidCharges per Year per Facility (for denominator)
total_charges = (
    drg_df.groupby(["Year", "FacilityName"])["ValidCharges"]
    .sum()
    .reset_index()
    .rename(columns={"ValidCharges": "TotalCharges"})
)


# ValidCharges per DRG per Year per Facility (numerator)
drg_charges = (
    drg_df.groupby(["Year", "FacilityName", "DRGDescription"])["ValidCharges"]
    .sum()
    .reset_index()
)


# Merge and calculate %
merged = pd.merge(drg_charges, total_charges, on=["Year", "FacilityName"])
merged["ChargePct"] = merged["ValidCharges"] / merged["TotalCharges"] * 100
merged["Year"] = merged["Year"].astype(str)


fig_drg = go.Figure()

for facility in selected_drg_facilities:
    for drg in selected_drgs:
        data = merged[
            (merged["FacilityName"] == facility) &
            (merged["DRGDescription"] == drg)
        ]
        if not data.empty:
            fig_drg.add_trace(
    go.Scatter(
        x=data["Year"],
        y=data["ValidCharges"],
        mode="lines+markers",
        name=f"{drg} - {facility}",
        hovertemplate=(
            rf"Year: %{{x}}<br>"
            rf"Facility: {facility}<br>"
            rf"DRG: {drg}<br>"
            rf"Valid Charges: $%{{y:,.0f}}<extra></extra>"
        )
    )
)



fig_drg.update_layout(
    title="Valid Charges Over Time by DRG and Facility",
    xaxis_title="Year",
    yaxis_title="Valid Charges ($)",
    xaxis=dict(type='category'),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.25,
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=40, r=40, t=80, b=160),
    height=500
)

st.plotly_chart(fig_drg, use_container_width=True)


#################################

df_payer["dsch_yr"] = df_payer["dsch_yr"].astype(str)  # for x-axis
df_payer["pay_cat1"] = df_payer["pay_cat1"].astype(str)
df_payer["patcnty1"] = df_payer["patcnty1"].astype(str)

counties = sorted(df_payer["patcnty1"].dropna().unique())
payers = sorted(df_payer["pay_cat1"].dropna().unique())

st.subheader("📊 Discharges Over Time by Payer")

selected_county = st.selectbox(
    "Select Patient County",
    options=counties,
    index=counties.index("Alameda") if "Alameda" in counties else 0
)

default_payers = [p for p in ["Medicare", "Private Coverage"] if p in payers]

selected_payers = st.multiselect(
    "Select Payers",
    options=payers,
    default=default_payers if default_payers else [payers[0]]
)


filtered_df = df_payer[
    (df_payer["patcnty1"] == selected_county) &
    (df_payer["pay_cat1"].isin(selected_payers))
]

grouped = (
    filtered_df
    .groupby(["dsch_yr", "pay_cat1"], as_index=False)["Discharges"]
    .sum()
)

# Plot
import plotly.graph_objects as go

fig = go.Figure()

import numpy as np

for payer in selected_payers:
    data = grouped[grouped["pay_cat1"] == payer].sort_values("dsch_yr")
    
    # Plot actual discharges
    fig.add_trace(
        go.Scatter(
            x=data["dsch_yr"],
            y=data["Discharges"],
            mode="lines+markers",
            name=payer,
            hovertemplate=(
                rf"Year: %{{x}}<br>Payer: {payer}<br>Discharges: %{{y:,}}<extra></extra>"
            )
        )
    )
    
    # Fit a trend line using polyfit (degree 1 = linear)
    if len(data) >= 2:
        x_numeric = data["dsch_yr"].astype(int)
        y_vals = data["Discharges"]
        coeffs = np.polyfit(x_numeric, y_vals, deg=1)
        trend_vals = np.polyval(coeffs, x_numeric)

        fig.add_trace(
            go.Scatter(
                x=data["dsch_yr"],
                y=trend_vals,
                mode="lines",
                name=f"{payer} Trend",
                line=dict(dash="dot"),
                hoverinfo="skip",
                showlegend=True
            )
        )


fig.update_layout(
    title=f"Discharges Over Time in {selected_county}",
    xaxis_title="Year",
    yaxis_title="Number of Discharges",
    xaxis=dict(type='category'),
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    height=500,
    margin=dict(l=40, r=40, t=80, b=140)
)


st.plotly_chart(fig, use_container_width=True)

###############################

import pandas as pd
import plotly.express as px

# Group by Title + Location + lat/lon to get counts
df_grouped = df_jobs.groupby(["Title", "Location", "lat", "lon"]).size().reset_index(name="Count")

# Create geo scatter plot
fig = px.scatter_geo(
    df_grouped,
    lat="lat",
    lon="lon",
    color="Title",
    size="Count",
    hover_name="Title",
    hover_data={"Location": True, "Count": True, "lat": False, "lon": False},
    scope="usa",  # 👈 Limit map to USA only
    title="📍 Hiring Locations (Kaiser) by Job Title",
)

fig.update_layout(
    height=1000,        # Increased height
    width=1600,        # Increased width (breadth)
    margin=dict(l=0, r=0, t=40, b=160),  # Reserve bottom space for legend
    legend=dict(
        orientation="h",
        x=0.5,
        y=-0.25,            # Keep legend well below the map
        xanchor="center",
        yanchor="top"
    ),
    showlegend=False  # 👈 this disables the legend
)

# Optional: refine bubble size scaling
fig.update_traces(marker=dict(sizemode="area", sizeref=2.*df_grouped["Count"].max()/(40.**2)))

# fig.show()

st.plotly_chart(fig, use_container_width=True)


# fig = px.scatter_mapbox(
#     df_grouped,
#     lat="lat",
#     lon="lon",
#     color="Title",
#     size="Count",
#     hover_name="Title",
#     hover_data={"Location": True, "Count": True, "lat": False, "lon": False},
#     zoom=3,  # Set zoom level for US view
#     mapbox_style="open-street-map",  # 👈 This enables OpenStreetMap base
#     title="📍 Hiring Locations (Kaiser) by Job Title",
#     height=1000
# )

# fig.update_layout(
#     width=1600,
#     margin=dict(l=0, r=0, t=40, b=0),
#     showlegend=False  # Optional: hide legend
# )

# st.plotly_chart(fig, use_container_width=True)




