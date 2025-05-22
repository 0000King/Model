import os
import pandas as pd
from datetime import datetime, timedelta
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage , AIMessage

# Google Gemini setup
google_api_key = "AIzaSyDx-N462cNd0r8aSMUeW3ufl-r0XD7abLY"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key,
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# Define persistent vector directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load Excel files
tasks_df = pd.read_excel("sugam/Sugam.xlsx", header=2)
timeline_df = pd.read_excel("sugam/Sugam Activity Timeline Report.xlsx", header=2)
attendance_df = pd.read_excel("sugam/Sugam attendance-summary-report.xlsx", header=2)

# Clean column names (remove whitespace)
tasks_df.columns = tasks_df.columns.str.strip()
timeline_df.columns = timeline_df.columns.str.strip()
attendance_df.columns = attendance_df.columns.str.strip()

# Helper function to process data
def process_shift_time(row):
    try:
        start = pd.to_datetime(row['First Mark In'])
        end = pd.to_datetime(row['Last Mark Out'])
        hours = (end - start).total_seconds() / 3600
        return min(hours, 9)
    except:
        return 0

def generate_text_chunks(tasks_df, timeline_df, attendance_df):
    documents = []

    # Process shift time
    timeline_df['Shift Hours'] = timeline_df.apply(process_shift_time, axis=1)

    # Melt attendance for daily view
    id_vars = ['Field Executive Username']
    date_columns = [col for col in attendance_df.columns if '/' in col]
    attendance_melted = attendance_df.melt(id_vars=id_vars, value_vars=date_columns,
                                           var_name="Date", value_name="Attendance")

    # Ensure date consistency and normalize (strip time)
    attendance_melted['Date'] = pd.to_datetime(attendance_melted['Date'], format="%d/%m/%Y", errors='coerce').dt.normalize()
    timeline_df['Date'] = pd.to_datetime(timeline_df['Date'], errors='coerce', dayfirst=True).dt.normalize()

    # --- UPDATED: Parse Actual Start Time with AM/PM format ---
    tasks_df['Actual Start Time'] = pd.to_datetime(
        tasks_df['Actual Start Time'],
        format="%d/%m/%Y %I:%M %p",  # 12-hour format with AM/PM
        errors='coerce'
    )
    # Normalize to date only (strip time)
    tasks_df['Actual Start Time Date'] = tasks_df['Actual Start Time'].dt.normalize()

    # Generate per-day, per-employee records
    for username in tasks_df['Field Executive Username'].unique():
        user_tasks = tasks_df[tasks_df['Field Executive Username'] == username]
        user_timeline = timeline_df[timeline_df['Field Executive Username'] == username]
        user_attendance = attendance_melted[attendance_melted['Field Executive Username'] == username]

        # Normalize the dates from user_tasks for iteration
        unique_dates = user_tasks['Actual Start Time Date'].dropna().unique()

        for date in unique_dates:
            day_tasks = user_tasks[user_tasks['Actual Start Time Date'] == date]
            timeline_row = user_timeline[user_timeline['Date'] == date]
            attendance_row = user_attendance[user_attendance['Date'] == date]

            name = day_tasks['Field Executive Name'].iloc[0] if not day_tasks.empty else "Unknown"
            team = day_tasks['Team'].iloc[0] if not day_tasks.empty else "Unknown"

            shift_hours = round(float(timeline_row['Shift Hours'].iloc[0]), 2) if not timeline_row.empty else "NA"
            attendance_status = attendance_row['Attendance'].iloc[0] if not attendance_row.empty else "NA"
            task_count = len(day_tasks)
            task_summary = "; ".join(day_tasks['Task Description'].fillna("No Description"))

            day_tasks['Task Duration'] = pd.to_numeric(day_tasks['Task Duration'], errors='coerce')
            duration_sum = day_tasks['Task Duration'].sum()
            distance_sum = day_tasks['Distance (KM)'].sum()

            content = f"""
Date: {date.date()}
Name: {name}
Team: {team}
Username: {username}
Shift Hours: {shift_hours}
Attendance: {attendance_status}
Tasks Completed: {task_count}
Task Duration: {duration_sum} hrs
Distance Covered: {distance_sum} KM
Task Details: {task_summary}
"""

            documents.append(Document(page_content=content))

    return documents

# Generate documents from raw Excel data
documents = generate_text_chunks(tasks_df, timeline_df, attendance_df)

# Load into ChromaDB
# db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persistent_directory)
# db.persist()
# for doc in documents:
#     if "Gaurav Sharma" in doc.page_content:
#         print(doc.page_content)