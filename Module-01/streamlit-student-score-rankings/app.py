import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Show the app title
st.title("Student Score Rankings")

# Upload a CSV or Excel file
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file is not None:
    # Read the uploaded file into a DataFrame
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Show the dataframe
    st.dataframe(df)

    # Only columns with "score" in the name are accepted
    st.write('Only columns that contain "score" can be selected.')
    score_columns = [column for column in df.columns if "score" in column.lower()]

    if len(score_columns) == 0:
        st.warning('No score column found. Please upload a file with a column containing "score".')
    else:
        # Let the user choose the score column
        score_column = st.selectbox("Select score column", score_columns)

        # Convert the selected column to numbers
        scores = pd.to_numeric(df[score_column])

        # Create a new column for score ranking
        df["ranking"] = "Failed"
        df.loc[scores >= 30, "ranking"] = "Poor"
        df.loc[scores >= 50, "ranking"] = "Normal"
        df.loc[scores >= 70, "ranking"] = "Good"
        df.loc[scores >= 90, "ranking"] = "Excellent"

        # Count students in each ranking
        total_students = len(df)
        ranking_order = ["Excellent", "Good", "Normal", "Poor", "Failed"]
        summary = df["ranking"].value_counts().reindex(ranking_order, fill_value=0).reset_index()
        summary.columns = ["ranking", "count"]
        summary["percentage"] = summary["count"] / total_students * 100

        # Show the summary table
        st.write("Ranking summary")
        st.dataframe(summary)

        # Plot student count by ranking
        chart_data = summary[summary["count"] > 0]
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#f1c40f", "#e74c3c"]
        fig, ax = plt.subplots()
        ax.pie(
            chart_data["count"],
            labels=chart_data["ranking"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors[: len(chart_data)],
        )
        ax.set_title("Student Count by Score Ranking")

        # Show the plot in Streamlit
        st.pyplot(fig)
else:
    # Message before a file is uploaded
    st.write("Upload student score to start. The file can be CSV or Excel.")
