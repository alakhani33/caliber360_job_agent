import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px
import re
import plotly.graph_objects as go
import google.generativeai as genai
import os

from dotenv import load_dotenv
# Load environment variables.
load_dotenv()

# Set the model name for our LLMs.
GEMINI_MODEL = "gemini-1.5-flash"
# Store the API key in a variable.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# def generate_executive_summary(data_description):
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     prompt = f"""
#     You are a healthcare strategy assistant. Based on the following data insights, generate a concise executive summary for hospital executives. Highlight key patterns, outliers, competitive insights, and strategic recommendations.

#     Data Summary:
#     {data_description}
    
#     Format:
#     1. Key Trends:
#     2. Competitive Positioning:
#     3. Strategic Risks/Opportunities:
#     4. Suggested Actions:
#     """

#     response = model.generate_content(prompt)
#     return response.text

@st.cache_data
def load_excel_data():
    
    # Example: load your data
    df_jobs = pd.read_csv("kaiser_sutter_jobs.csv")
    return df_jobs

df_jobs = load_excel_data()


# st.title("🤖 CALIBER360 Job Strategist AI Agent")

# st.markdown(
#     "<h4 style='margin-top: -10px; color: blue;'>Competitive Hiring Insights at your fingertips</h4>",
#     unsafe_allow_html=True
# )
st.title("🤖 CALIBER360 Job Strategist AI Agent")

st.markdown(
    """
    <h4 style='text-align: center; margin-top: -10px; color: #2E86C1; font-weight: 500;'>
        Competitive Hiring Insights at your Fingertips
    </h4>
    """,
    unsafe_allow_html=True
)


import streamlit as st
import pandas as pd
import plotly.express as px

# Split the job data
kaiser_jobs = df_jobs[df_jobs['Facility'] == 'Kaiser Permanente']
sutter_jobs = df_jobs[df_jobs['Facility'] == 'Sutter Health']

def prepare_map_data(df):
    df_clean = df.dropna(subset=["lat", "lon"])
    df_grouped = (
        df_clean.groupby(["Title", "Location", "lat", "lon"])
        .size()
        .reset_index(name="Count")
    )
    return df_grouped

def create_plot(df_grouped, title):
    fig = px.scatter_geo(
        df_grouped,
        lat="lat",
        lon="lon",
        color="Title",
        size="Count",
        hover_name="Title",
        hover_data={"Location": True, "Count": True, "lat": False, "lon": False},
        scope="usa",
        title=title,
    )

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=40, b=40),
        showlegend=False
    )

    fig.update_traces(marker=dict(sizemode="area", sizeref=2. * df_grouped["Count"].max() / (40. ** 2)))
    return fig

# Prepare the data
kaiser_map_data = prepare_map_data(kaiser_jobs)
sutter_map_data = prepare_map_data(sutter_jobs)

# Create maps
kaiser_fig = create_plot(kaiser_map_data, "📍 Kaiser Permanente")
sutter_fig = create_plot(sutter_map_data, "📍 Sutter Health")

# Display side by side
st.subheader("🗺️ Job Postings")
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(kaiser_fig, use_container_width=True)

with col2:
    st.plotly_chart(sutter_fig, use_container_width=True)


import google.generativeai as genai

# 🔑 Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

# 🧠 Prepare the input summary (actual data)
def generate_job_summary(df, facility_name):
    top_roles = (
        df["Title"]
        .value_counts()
        # .head(5)
        .to_frame("Count")
        .reset_index()
        .rename(columns={"index": "Title"})
    )

    top_locations = (
        df["Location"]
        .value_counts()
        # .head(5)
        .to_frame("Count")
        .reset_index()
        .rename(columns={"index": "Location"})
    )

    summary = f"\n=== {facility_name} ===\n"
    summary += f"Total Jobs: {len(df)}\n"
    summary += "Top Roles:\n"
    for _, row in top_roles.iterrows():
        summary += f"- {row['Title']} ({row['Count']})\n"
    summary += "Top Locations:\n"
    for _, row in top_locations.iterrows():
        summary += f"- {row['Location']} ({row['Count']})\n"
    return summary


import google.generativeai as genai

# Ensure Gemini is configured
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Filter job data for Kaiser and Sutter
kaiser_df = df_jobs[df_jobs["Facility"] == "Kaiser Permanente"]
sutter_df = df_jobs[df_jobs["Facility"] == "Sutter Health"]

# Optional: further clean or reduce job titles for LLM input
def format_job_summary(df, name):
    summary = f"\n{name} Job Postings:\n"
    grouped = (
        df.groupby(["Title", "Location"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .head(25)  # limit for brevity
    )
    for _, row in grouped.iterrows():
        summary += f" - {row['Title']} in {row['Location']} ({row['Count']} postings)\n"
    return summary

# Create prompt
kaiser_summary = format_job_summary(kaiser_df, "Kaiser Permanente")
sutter_summary = format_job_summary(sutter_df, "Sutter Health")

prompt = f"""
You're a healthcare workforce strategy advisor. Based on these recent job postings from Kaiser Permanente and Sutter Health, provide a comparative analysis including:

1. Most prominent job categories by system.
2. Key geographic differences in hiring.
3. Notable or emerging roles.
4. Strategic implications for competitors in the region.

{kaiser_summary}
{sutter_summary}

Format your response clearly and insightfully.
"""


st.markdown("---")
st.subheader("🧠 CALIBER360 - Executive Summary")

# Set up Gemini chat model (if not already initialized)
if "job_chat_session" not in st.session_state:
    st.session_state.job_chat_session = model.start_chat(history=[])

# Automatically generate the summary if not already stored
if "job_ai_summary" not in st.session_state or st.button("🔁 Regenerate Summary"):
    with st.spinner("Analyzing hiring data..."):
        response = st.session_state.job_chat_session.send_message(prompt)
        st.session_state.job_ai_summary = response.text

# Display the summary
st.markdown(f"""
<div style="border: 2px solid #ccc; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
  <p style="font-size: 16px; line-height: 1.6;">{st.session_state.job_ai_summary}</p>
</div>
""", unsafe_allow_html=True)

# Suggested follow-up questions
sample_questions = [
    "What types of roles are being hired the most?",
    "How do the hiring patterns differ between Kaiser and Sutter?",
    "Are there any geographic hotspots for hiring?",
    "What strategic goals might be reflected in this hiring?",
    "What advice would you give to a competitor like UC Davis Health?"
]

# st.markdown("💬 **Ask a follow-up question about the hiring trends:**")
st.markdown("---")
st.subheader("🧠 Ask a follow-up question about the hiring trends:")

# # Show sample suggestions
# with st.expander("💡 Sample Questions You Can Ask"):
#     for q in sample_questions:
#         st.markdown(f"• _{q}_")
st.markdown("""
<div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; border: 1px solid #d3d3d3;'>
    <h5 style='margin-bottom: 10px; color: #2c3e50;'>💬 Sample Questions You Can Ask</h5>
    <ul style='margin-left: 20px; color: #34495e;'>
        <li>What advice would you give to a competitor like UC Davis Health?</li>
        <li>What strategic goals might be reflected in this hiring?</li>
        <li>What implications does this have for competitors?</li>
        <li>Which locations have the highest job concentration?</li>
        <li>How do hiring patterns reflect strategic growth?</li>
        <li>How do the hiring patterns differ between Kaiser and Sutter?</li>
    </ul></div>
""", unsafe_allow_html=True)


# Input field for user question
user_question = st.text_input("Your question")

if user_question:
    with st.spinner("Thinking..."):
        response = st.session_state.job_chat_session.send_message(user_question)
        followup_answer = response.text

    # st.markdown(f"""
    # <div style="border-left: 5px solid #4CAF50; padding: 10px; margin-top: 10px; background-color: #eef9f0;">
    # <strong>CALIBER360 AI Response:</strong><br>{followup_answer}
    # </div>
    # """, unsafe_allow_html=True)

    # import html

    # st.markdown(f"""
    #     <div style="border-left: 5px solid #4CAF50; padding: 10px; margin-top: 10px; background-color: #eef9f0;">
    #     <strong>CALIBER360 AI Response:</strong><br>{html.escape(followup_answer).replace('\\n', '<br>')}
    #     </div>
    # """, unsafe_allow_html=True)
    import html

    escaped_answer = html.escape(followup_answer)

    st.markdown(f"""
        <div style="border-left: 5px solid #4CAF50; padding: 10px; margin-top: 10px; background-color: #eef9f0;">
        <strong>CALIBER360 AI Response:</strong><br>{escaped_answer}</div>
    """, unsafe_allow_html=True)


    st.markdown("---")

    


st.markdown(
    "<p style='font-size: 20px; color: #444;'>"
    "💡 <strong>Every dashboard can be made intelligent</strong> by adding an AI Agent. "
    "If you're ready to supercharge your insights, we can help. "
    "<a href='https://caliber360ai.com' target='_blank'>Reach out to us at caliber360ai.com</a>."
    "</p>",
    unsafe_allow_html=True
)


