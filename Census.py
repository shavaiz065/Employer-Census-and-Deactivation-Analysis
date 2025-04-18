import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np


# ---------- CONFIG ----------
st.set_page_config(page_title="Census & Deactivation Dashboard", layout="wide")

# ---------- THEMING ----------
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark", "Corporate"])

# Custom styling based on the theme
if theme == "Dark":
    st.markdown("""
    <style>
        body { background-color: #0e1117; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .stSidebar {background-color: #1d2025;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; font-size: 16px;}
        .stMetric>div {color: #ff7043;}
    </style>
    """, unsafe_allow_html=True)
elif theme == "Corporate":
    st.markdown("""
    <style>
        body { background-color: #f1f5f9; color: #1e293b; font-family: 'Arial', sans-serif; }
        .stSidebar {background-color: #f8fafc;}
        .stButton>button {background-color: #0288d1; color: white; border-radius: 8px; font-size: 16px;}
    </style>
    """, unsafe_allow_html=True)


# ---------- FILE UPLOAD ----------
with st.sidebar:
    st.header("Upload Files")
    census_file = st.file_uploader("Census Processing Report", type=["csv", "xlsx"])
    deact_file = st.file_uploader("Deactivation Report", type=["csv", "xlsx"])

# ---------- LOAD DATA ----------
def load_data(uploaded_file):
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Convert date columns to datetime
        if 'processendtime' in df.columns:
            df['processendtime'] = pd.to_datetime(df['processendtime'], errors='coerce')
        if 'processstarttime' in df.columns:
            df['processstarttime'] = pd.to_datetime(df['processstarttime'], errors='coerce')
        if 'recorddate' in df.columns:
            df['recorddate'] = pd.to_datetime(df['recorddate'], errors='coerce')

        # Convert numeric columns to numeric type
        if 'completed' in df.columns:
            df['completed'] = pd.to_numeric(df['completed'], errors='coerce')
        if 'error' in df.columns:
            df['error'] = pd.to_numeric(df['error'], errors='coerce')
        if 'filerecordbeforeprocessing' in df.columns:
            df['filerecordbeforeprocessing'] = pd.to_numeric(df['filerecordbeforeprocessing'], errors='coerce')

        return df
    return pd.DataFrame()

census_df = load_data(census_file)
deact_df = load_data(deact_file)

# ---------- STANDARDIZE COLUMNS ----------
def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip().str.lower()  # Ensure columns are strings
    return df

census_df = clean_columns(census_df)
deact_df = clean_columns(deact_df)

# ---------- UI TABS ----------
tabs = st.tabs(["📋 Overview", "📅 Date Insights", "🔍 Employer Drilldown", "⚠️ Alerts", "📊 In-System Variance"])

# ---------- TAB 1: OVERVIEW ----------
with tabs[0]:
    st.title("📋 Overview")

    if not census_df.empty and not deact_df.empty:
        # Convert datetime columns
        census_df['processstarttime'] = pd.to_datetime(census_df['processstarttime'], errors='coerce')
        census_df['processendtime'] = pd.to_datetime(census_df['processendtime'], errors='coerce')
        deact_df['recorddate'] = pd.to_datetime(deact_df['recorddate'], errors='coerce')

        # ------ FILTERS SECTION ------
        st.subheader("Filters")

        # Date range filter
        census_dates = census_df['processendtime'].dropna()
        deact_dates = deact_df['recorddate'].dropna()

        if not census_dates.empty and not deact_dates.empty:
            min_date = min(census_dates.min(), deact_dates.min()).date()
            max_date = max(census_dates.max(), deact_dates.max()).date()
        else:
            min_date = datetime.now().date() - timedelta(days=30)
            max_date = datetime.now().date()

        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )

        # Employer filter
        with col2:
            census_employers = census_df['employer'].unique() if 'employer' in census_df.columns else []
            deact_employer_col = 'employername' if 'employername' in deact_df.columns else 'employer'
            deact_employers = deact_df[deact_employer_col].unique() if deact_employer_col in deact_df.columns else []

            all_employers = sorted(list(set(census_employers).union(set(deact_employers))))
            selected_employer = st.selectbox(
                "Filter by Employer",
                options=["All Employers"] + all_employers
            )

        # Apply date filtering
        if len(date_range) == 2:
            start_date, end_date = date_range
            census_filtered = census_df[
                (census_df['processendtime'].dt.date >= start_date) &
                (census_df['processendtime'].dt.date <= end_date)
                ]
            deact_filtered = deact_df[
                (deact_df['recorddate'].dt.date >= start_date) &
                (deact_df['recorddate'].dt.date <= end_date)
                ]
        else:
            census_filtered = census_df.copy()
            deact_filtered = deact_df.copy()

        # Apply employer filtering
        if selected_employer != "All Employers":
            census_filtered = census_filtered[census_filtered['employer'] == selected_employer]
            deact_filtered = deact_filtered[deact_filtered[deact_employer_col] == selected_employer]

        # ------ SUMMARY STATISTICS ------
        st.subheader("📊 Summary Statistics")

        # Calculate stats on filtered data
        total_records = len(census_filtered)
        total_errors = census_filtered['error'].notna().sum() if 'error' in census_filtered.columns else 0
        total_completed = census_filtered['completed'].notna().sum() if 'completed' in census_filtered.columns else 0
        total_completed_percentage = (total_completed / total_records * 100) if total_records > 0 else 0

        census_filtered['processing_time_mins'] = (
                (census_filtered['processendtime'] - census_filtered['processstarttime']).dt.total_seconds() / 60
        )
        avg_processing_time = census_filtered['processing_time_mins'].mean()

        # Display stats
        st.write(f"Total Records Processed: {total_records}")
        st.write(f"Total Errors: {total_errors}")
        st.write(f"Percentage Completed: {total_completed_percentage:.2f}%")
        st.write(f"Avg Processing Time: {avg_processing_time:.2f} minutes")

        # ------ LATEST PROCESSING TABLE ------
        st.subheader("Latest Processing by Employer")

        if not census_filtered.empty and 'employer' in census_filtered.columns:
            latest_processing = census_filtered.groupby('employer').agg({
                'processendtime': 'max',
                'filerecordbeforeprocessing': 'last',
                'completed': 'last',
                'error': 'last'
            }).reset_index()


            # Formatting function
            def format_value(x):
                if pd.isna(x):
                    return 'N/A'
                try:
                    x = float(x)
                    return f"{x:,.0f}" if x == int(x) else f"{x:,.2f}"
                except (ValueError, TypeError):
                    return str(x)


            # Display table
            st.dataframe(
                latest_processing.style.format({
                    'processendtime': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else 'N/A',
                    'filerecordbeforeprocessing': format_value,
                    'completed': format_value,
                    'error': format_value
                }),
                use_container_width=True
            )

        # ------ PROCESSING WITH DEACTIVATIONS TABLE ------
        st.subheader("Recent Processing with Deactivations")

        if not census_filtered.empty and not deact_filtered.empty:
            # Get all processing records within date range (not just latest)
            census_within_range = census_filtered[
                (census_filtered['processendtime'].dt.date >= start_date) &
                (census_filtered['processendtime'].dt.date <= end_date)
                ]

            # Get deactivations within date range
            deact_within_range = deact_filtered[
                (deact_filtered['recorddate'].dt.date >= start_date) &
                (deact_filtered['recorddate'].dt.date <= end_date)
                ]

            # Merge processing with deactivations on same date
            merged = pd.merge(
                census_within_range,
                deact_within_range,
                left_on=['employer', census_within_range['processendtime'].dt.date],
                right_on=[deact_employer_col, deact_within_range['recorddate'].dt.date],
                how='left'
            )

            # Select and rename columns
            result_cols = [
                'employer', 'processendtime', 'filerecordbeforeprocessing',
                'completed', 'error', 'deactivations'
            ]
            result_df = merged[result_cols].rename(columns={
                'processendtime': 'Process Time',
                'filerecordbeforeprocessing': 'Records',
                'completed': 'Completed',
                'error': 'Errors',
                'deactivations': 'Deactivations'
            })

            # Convert numeric columns
            numeric_cols = ['Records', 'Completed', 'Errors', 'Deactivations']
            for col in numeric_cols:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

            # Format datetime
            result_df['Process Time'] = result_df['Process Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Sort by employer and process time
            result_df = result_df.sort_values(['employer', 'Process Time'])

            # Display table
            st.dataframe(
                result_df.style.format({
                    'Records': '{:,.0f}',
                    'Completed': '{:,.0f}',
                    'Errors': '{:,.0f}',
                    'Deactivations': '{:,.0f}'
                }),
                use_container_width=True
            )

        # ------ DOWNLOAD BUTTONS ------
        st.subheader("Export Data")
        col1, col2 = st.columns(2)

        with col1:
            if not census_filtered.empty:
                st.download_button(
                    label="Download Processing Data",
                    data=census_filtered.to_csv(index=False),
                    file_name="processing_data.csv",
                    mime="text/csv"
                )

        with col2:
            if not deact_filtered.empty:
                st.download_button(
                    label="Download Deactivation Data",
                    data=deact_filtered.to_csv(index=False),
                    file_name="deactivation_data.csv",
                    mime="text/csv"
                )

    else:
        st.info("Please upload both reports to view this section.")

# ---------- TAB 2: DATE INSIGHTS ----------
with tabs[1]:
    st.title("📅 Date-Wise Insights")
    if not census_df.empty and not deact_df.empty:
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", value=datetime.today())

        # Ensure date columns are datetime and numeric columns are numeric
        census_df['processstarttime'] = pd.to_datetime(census_df['processstarttime'])
        deact_df['recorddate'] = pd.to_datetime(deact_df['recorddate'])

        # Convert numeric columns to numeric type
        for col in ['filerecordbeforeprocessing', 'completed', 'error']:
            if col in census_df.columns:
                census_df[col] = pd.to_numeric(census_df[col], errors='coerce').fillna(0)

        for col in ['deactivations']:
            if col in deact_df.columns:
                deact_df[col] = pd.to_numeric(deact_df[col], errors='coerce').fillna(0)

        # Filter data for selected date range
        df_census_range = census_df[
            (census_df['processstarttime'].dt.date >= start_date) &
            (census_df['processstarttime'].dt.date <= end_date)
            ]
        df_deact_range = deact_df[
            (deact_df['recorddate'].dt.date >= start_date) &
            (deact_df['recorddate'].dt.date <= end_date)
            ]

        # Create tabs for different views
        view_tabs = st.tabs(["Daily Summary", "Detailed View"])

        with view_tabs[0]:
            st.subheader(f"Summary for {start_date} to {end_date}")

            # Summary metrics for the date range
            col1, col2, col3 = st.columns(3)
            with col1:
                total_files = int(
                    df_census_range['filerecordbeforeprocessing'].sum()) if not df_census_range.empty else 0
                st.metric("Total Files Processed", total_files)
            with col2:
                total_completed = int(df_census_range['completed'].sum()) if not df_census_range.empty else 0
                st.metric("Total Completed", total_completed)
            with col3:
                total_errors = int(df_census_range['error'].sum()) if not df_census_range.empty else 0
                st.metric("Total Errors", total_errors)

            # Daily breakdown chart
            st.subheader("Daily Deactivations Trend")
            if not df_deact_range.empty:
                daily_deact = df_deact_range.groupby(df_deact_range['recorddate'].dt.date)[
                    'deactivations'].sum().reset_index()
                daily_deact.columns = ['date', 'deactivations']

                fig_daily = px.line(
                    daily_deact,
                    x='date',
                    y='deactivations',
                    title=f"Daily Deactivations from {start_date} to {end_date}",
                    markers=True
                )
                st.plotly_chart(fig_daily, use_container_width=True)

                # 7-day moving average
                trend = df_deact_range.copy()
                trend_ma = trend.groupby(trend['recorddate'].dt.date)['deactivations'].sum().rolling(7,
                                                                                                     min_periods=1).mean()
                avg_7 = trend_ma.mean()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("7-Day Moving Average", f"{avg_7:.2f}" if not np.isnan(avg_7) else "N/A")
                with col2:
                    total_deact = int(df_deact_range['deactivations'].sum())
                    st.metric("Total Deactivations in Selected Range", total_deact)
            else:
                st.info("No deactivation data available for the selected date range.")

        with view_tabs[1]:
            st.subheader(f"Detailed Data for {start_date} to {end_date}")

            # Census Data Table
            st.markdown("**Census Data**")
            if not df_census_range.empty:
                census_pivot = df_census_range.pivot_table(
                    index='employer',
                    values=['filerecordbeforeprocessing', 'completed', 'error'],
                    aggfunc='sum'
                ).reset_index()
                # Convert to integer for cleaner display
                for col in ['filerecordbeforeprocessing', 'completed', 'error']:
                    if col in census_pivot.columns:
                        census_pivot[col] = census_pivot[col].astype(int)
                st.dataframe(census_pivot)
            else:
                st.info("No census data available for the selected date range.")

            # Deactivations Data Table
            st.markdown("**Deactivation Data**")
            if not df_deact_range.empty:
                deact_pivot = df_deact_range.pivot_table(
                    index='employername',
                    values='deactivations',
                    aggfunc='sum'
                ).reset_index()
                deact_pivot['deactivations'] = deact_pivot['deactivations'].astype(int)
                st.dataframe(deact_pivot)
            else:
                st.info("No deactivation data available for the selected date range.")

            # Download buttons for the filtered data
            col1, col2 = st.columns(2)
            with col1:
                if not df_census_range.empty:
                    csv_census = df_census_range.to_csv(index=False)
                    st.download_button(
                        label="Download Census Data",
                        data=csv_census,
                        file_name=f"census_data_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    )
            with col2:
                if not df_deact_range.empty:
                    csv_deact = df_deact_range.to_csv(index=False)
                    st.download_button(
                        label="Download Deactivation Data",
                        data=csv_deact,
                        file_name=f"deactivation_data_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("Please upload both reports to view this section.")

# ---------- TAB 3: EMPLOYER DRILLDOWN ----------
with tabs[2]:
    st.title("🔍 Employer Drilldown")
    if not census_df.empty and not deact_df.empty:
        employer_list = sorted(census_df['employer'].dropna().unique())
        employer_selected = st.selectbox("Select Employer", ["All Employers"] + employer_list)

        if employer_selected != "All Employers":
            emp_census = census_df[census_df['employer'] == employer_selected]
            emp_deact = deact_df[deact_df['employername'] == employer_selected]

            st.subheader("📈 Processing Pattern")
            chart = px.scatter(emp_census, x='processstarttime', y='completed', color='error',
                               title=f"Processing History for {employer_selected}")
            st.plotly_chart(chart, use_container_width=True)

            emp_census['processing_time_mins'] = (emp_census['processendtime'] - emp_census['processstarttime']).dt.total_seconds() / 60
            st.write(emp_census[['processstarttime', 'processendtime', 'processing_time_mins']])

            st.subheader("📉 Deactivation Trend")
            if not emp_deact.empty:
                emp_deact['recorddate'] = pd.to_datetime(emp_deact['recorddate'])
                line = px.line(emp_deact, x='recorddate', y='deactivations')
                st.plotly_chart(line, use_container_width=True)
        else:
            st.info("Please select a specific employer to view detailed data.")
    else:
        st.info("Please upload both reports to view this section.")

# ---------- TAB 4: ALERTS ----------
with tabs[3]:
    st.title("⚠️ Alerts & Exceptions")
    if not census_df.empty and not deact_df.empty:
        today = datetime.today().date()

        # Convert 'processendtime' to datetime, invalid entries will become NaT
        census_df['processendtime'] = pd.to_datetime(census_df['processendtime'], errors='coerce')

        # If there are any NaT (invalid date) values, display a warning
        if census_df['processendtime'].isnull().any():
            st.warning("Some 'processendtime' values were invalid and have been set to NaT (Not a Time).")

        # Drop rows with NaT values in 'processendtime' (invalid date)
        census_df = census_df.dropna(subset=['processendtime'])

        # Group by employer and get the latest processing date
        if 'processendtime' in census_df.columns:
            census_df['processendtime'] = pd.to_datetime(census_df['processendtime'], errors='coerce')

        # Drop NaT values to avoid issues with max operation
        valid_census_df = census_df.dropna(
            subset=['processendtime']) if 'processendtime' in census_df.columns else census_df

        # Then perform the groupby operation
        if not valid_census_df.empty and 'employer' in valid_census_df.columns:
            alerts = valid_census_df.groupby('employer')['processendtime'].max().reset_index()
        else:
            alerts = pd.DataFrame(columns=['employer', 'processendtime'])  # Create empty DataFrame with columns

        # Calculate the number of days since the last processing
        alerts['days_since_last'] = (datetime.now() - alerts['processendtime']).dt.days

        # Flag employers whose files weren't processed in the last 3 days
        flagged = alerts[alerts['days_since_last'] > 3]

        # Display a warning and a table with the flagged employers
        st.warning(f"Employers not processed in the last 3 days: {len(flagged)}")
        st.dataframe(flagged)
    else:
        st.info("Please upload both reports to view this section.")

# ---------- TAB 5: IN-SYSTEM VARIANCE ----------
with tabs[4]:
    st.title("📊 In-System Variance")
    if census_df.empty or deact_df.empty:
        st.info("Please upload both reports to view this section.")
        st.stop()

    # Convert date columns to datetime and numeric columns to numeric
    census_df['processstarttime'] = pd.to_datetime(census_df['processstarttime'], errors='coerce')
    deact_df['recorddate'] = pd.to_datetime(deact_df['recorddate'], errors='coerce')

    # Convert numeric columns
    for col in ['filerecordbeforeprocessing', 'completed', 'error']:
        if col in census_df.columns:
            census_df[col] = pd.to_numeric(census_df[col], errors='coerce')

    if 'deactivations' in deact_df.columns:
        deact_df['deactivations'] = pd.to_numeric(deact_df['deactivations'], errors='coerce')

    # Filter out rows with NaN dates
    valid_census_df = census_df.dropna(subset=['processstarttime'])
    valid_deact_df = deact_df.dropna(subset=['recorddate'])

    if valid_census_df.empty or valid_deact_df.empty:
        st.error("No valid date data found in one or both reports.")
        st.stop()

    # Get date range of data
    census_min_date = valid_census_df['processstarttime'].min().date()
    census_max_date = valid_census_df['processstarttime'].max().date()
    deact_min_date = valid_deact_df['recorddate'].min().date()
    deact_max_date = valid_deact_df['recorddate'].max().date()

    # Use the common date range
    min_date = max(census_min_date, deact_min_date)
    max_date = min(census_max_date, deact_max_date)

    st.write(f"Data date range: {min_date} to {max_date}")

    # Create filters
    col1, col2 = st.columns(2)

    with col1:
        # Date filter
        selected_date = st.date_input(
            "Select date:",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

    # Get all unique employers from both dataframes
    census_employers = valid_census_df['employer'].unique()
    deact_employers = valid_deact_df['employername'].unique()
    all_employers = sorted(list(set(census_employers) | set(deact_employers)))

    with col2:
        # Employer filter
        selected_employer = st.selectbox(
            "Filter by Employer:",
            ["All Employers"] + all_employers
        )

    # Filter data by date
    census_day_data = valid_census_df[valid_census_df['processstarttime'].dt.date == selected_date].copy()
    deact_day_data = valid_deact_df[valid_deact_df['recorddate'].dt.date == selected_date].copy()

    # Additional filter by employer if selected
    if selected_employer != "All Employers":
        census_day_data = census_day_data[census_day_data['employer'] == selected_employer]
        deact_day_data = deact_day_data[deact_day_data['employername'] == selected_employer]

    # Aggregate data by employer
    census_agg = census_day_data.groupby('employer').agg({
        'filerecordbeforeprocessing': 'sum',
        'completed': 'sum',
        'error': 'sum'
    }).reset_index()

    # Aggregate deactivation data by employer
    deact_agg = deact_day_data.groupby('employername').agg({
        'deactivations': 'sum'
    }).reset_index()
    # Rename column for consistency
    deact_agg = deact_agg.rename(columns={'employername': 'employer'})

    # Create and merge the comparison dataframe
    if census_agg.empty and deact_agg.empty:
        st.info(f"No data available for {selected_date}")
        st.stop()

    # Get all unique employers from filtered data
    filtered_employers = set(census_agg['employer'].unique() if not census_agg.empty else [])
    filtered_employers.update(set(deact_agg['employer'].unique() if not deact_agg.empty else []))

    # Create base dataframe with all relevant employers
    comparison_df = pd.DataFrame({'employer': list(filtered_employers)})

    # Merge data
    comparison_df = pd.merge(comparison_df, census_agg, on='employer', how='left')
    comparison_df = pd.merge(comparison_df, deact_agg, on='employer', how='left')

    # Fill NaN values with zeros
    for col in ['filerecordbeforeprocessing', 'completed', 'error', 'deactivations']:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].fillna(0)

    # Calculate percentage of deactivations
    if 'completed' in comparison_df.columns and 'deactivations' in comparison_df.columns:
        import numpy as np

        comparison_df['deactivation_percentage'] = np.where(
            comparison_df['completed'] > 0,
            (comparison_df['deactivations'] / comparison_df['completed']) * 100,
            0
        )

    # Display the comparison table
    st.subheader(f"Census Processing vs Deactivations on {selected_date}")

    # Format for display
    display_df = comparison_df.copy()
    if 'deactivation_percentage' in display_df.columns:
        display_df['deactivation_percentage'] = display_df['deactivation_percentage'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        )

    st.dataframe(display_df, use_container_width=True)

    # Download button
    st.download_button(
        "Download Variance Report",
        comparison_df.to_csv(index=False),
        file_name=f"variance_report_{selected_date}.csv",
        mime="text/csv",
        key="download_variance_report"
    )

    # Display metrics
    st.subheader("Visualizations")

    # Calculate totals
    metrics = {
        "Total Records Processed": comparison_df['filerecordbeforeprocessing'].sum(),
        "Total Completed": comparison_df['completed'].sum(),
        "Total Errors": comparison_df['error'].sum(),
        "Total Deactivations": comparison_df['deactivations'].sum()
    }

    # Show metrics
    cols = st.columns(4)
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(label, f"{value:,.0f}")

    # Create visualizations if we have more than one employer
    if len(comparison_df) > 1:
        chart_col1, chart_col2 = st.columns(2)

        # Chart 1: Processing vs Deactivations
        with chart_col1:
            fig = px.bar(
                comparison_df,
                x='employer',
                y=['completed', 'deactivations'],
                title='Completed Records vs Deactivations by Employer',
                barmode='group',
                labels={'value': 'Count', 'employer': 'Employer', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Chart 2: Deactivation Percentage
        with chart_col2:
            if 'deactivation_percentage' in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x='employer',
                    y='deactivation_percentage',
                    title='Deactivation Percentage by Employer',
                    color='deactivation_percentage',
                    color_continuous_scale=['green', 'yellow', 'red'],
                    labels={'deactivation_percentage': 'Deactivation %', 'employer': 'Employer'}
                )
                st.plotly_chart(fig, use_container_width=True)

    # Time series analysis for "All Employers" view
    if selected_employer == "All Employers":
        st.subheader("Time Series Analysis")

        # Define date range
        end_date = selected_date
        start_date = end_date - pd.Timedelta(days=30)  # Show last 30 days

        # Filter data for time series
        census_time_series = valid_census_df[
            (valid_census_df['processstarttime'].dt.date >= start_date) &
            (valid_census_df['processstarttime'].dt.date <= end_date)
            ].copy()

        deact_time_series = valid_deact_df[
            (valid_deact_df['recorddate'].dt.date >= start_date) &
            (valid_deact_df['recorddate'].dt.date <= end_date)
            ].copy()

        # Aggregate by date
        census_by_date = census_time_series.groupby(census_time_series['processstarttime'].dt.date).agg({
            'completed': 'sum',
            'error': 'sum'
        }).reset_index()

        deact_by_date = deact_time_series.groupby(deact_time_series['recorddate'].dt.date).agg({
            'deactivations': 'sum'
        }).reset_index()

        # Merge time series data
        time_series_df = pd.merge(
            census_by_date,
            deact_by_date,
            left_on='processstarttime',
            right_on='recorddate',
            how='outer'
        ).fillna(0)

        # Create time series chart
        if not time_series_df.empty:
            fig = px.line(
                time_series_df,
                x='processstarttime',
                y=['completed', 'deactivations'],
                title='Completed Records vs Deactivations Over Time',
                labels={'processstarttime': 'Date', 'value': 'Count', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)
