"""
Data Analyst Agent Web Application.

Streamlit-based interface for comprehensive data analysis including extraction, cleaning, and ML analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import analyze_dataframe
from llm_suggester import get_cleaning_suggestions
from code_generator import generate_python_code_from_final_instructions
from executor import execute_cleaning_code
from data_analysis import DataAnalyzer
from history_manager import save_cleaning_session, get_history
import logging

# Configure page
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
logger = logging.getLogger(__name__)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Data Cleaning", "Data Analysis"])

st.sidebar.markdown("---")
st.sidebar.markdown("### How it works:")
st.sidebar.markdown("""
1. **Data Cleaning**: Upload, clean, and prepare your data
2. **Data Analysis**: Analyze cleaned data with ML & insights
""")


def apply_custom_style():
    css = """
    <style>
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
        color: #212121;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        border-radius: 0.75rem;
        padding: 0.85rem 1.1rem;
        font-weight: 600;
        background-color: #2563eb;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .stMetric {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(200,200,200,0.4);
        border-radius: 1rem;
        padding: 1rem;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #111827;
        font-weight: 700;
    }
    .stMarkdown p {
        color: #4b5563;
        line-height: 1.6;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 0.75rem;
        border: 1px solid rgba(148,163,184,0.35);
    }
    .css-1d391kg {padding: 1.5rem 1rem;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_custom_style()

def show_data_cleaning_page():
    """Data Cleaning Page - Upload, clean, and prepare data"""
    st.title("Data Cleaning Studio")
    st.markdown("Upload your dataset, get AI-powered cleaning suggestions, and prepare your data for analysis.")

    # --- Upload Dataset ---
    st.header("Step 1: Upload Dataset")
    uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xls", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

        st.success(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Store in session state
        st.session_state['original_df'] = df
        st.session_state['current_df'] = df.copy()

        # --- Data Preview ---
        st.header("Step 2: Data Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        # Data preview tabs
        tab1, tab2, tab3 = st.tabs(["Data Head", "Data Info", "Missing Values"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            buffer = []
            buffer.append(f"**Shape:** {df.shape}")
            buffer.append(f"**Columns:** {', '.join(df.columns.tolist())}")
            buffer.append(f"**Data Types:**")
            for col, dtype in df.dtypes.items():
                buffer.append(f"  - {col}: {dtype}")
            st.markdown("\n".join(buffer))
        with tab3:
            missing_data = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)

        # --- AI Cleaning Suggestions ---
        st.header("Step 3: AI Cleaning Suggestions")

        # Generate summary for AI
        summary = analyze_dataframe(df)

        with st.spinner(" Analyzing your data and generating cleaning suggestions..."):
            suggestions = get_cleaning_suggestions(summary)

        # Parse and format suggestions as bullet points
        st.subheader("AI Recommendations:")

        # Split suggestions into bullet points
        suggestion_lines = [line.strip() for line in suggestions.split('\n') if line.strip()]

        for i, suggestion in enumerate(suggestion_lines, 1):
            if suggestion.startswith('-') or suggestion.startswith('•'):
                st.markdown(f"**{i}.** {suggestion[1:].strip()}")
            elif any(keyword in suggestion.lower() for keyword in ['handle', 'remove', 'fill', 'convert', 'clean', 'fix']):
                st.markdown(f"**{i}.** {suggestion}")
            else:
                st.markdown(f"**{i}.** {suggestion}")

        # --- User Customization ---
        st.header("Step 4: Customize Instructions")
        st.markdown("Add your own cleaning requirements or modify the AI suggestions:")

        user_instructions = st.text_area(
            "Additional Instructions (Optional)",
            placeholder="Example: Convert all column names to lowercase, or remove rows where age < 0",
            height=100
        )

        # Combine instructions
        final_instructions = suggestions
        if user_instructions.strip():
            final_instructions += f"\n\n Additional User Instructions:\n{user_instructions.strip()}"

        # --- Code Generation ---
        st.header("Step 5: Generate Cleaning Code")

        if st.button("Generate Python Code", type="primary", use_container_width=True):
            with st.spinner("Generating optimized cleaning code..."):
                code = generate_python_code_from_final_instructions(final_instructions)
                st.session_state['cleaning_code'] = code

            st.success("Code generated successfully!")

            # Display code with syntax highlighting
            st.code(code, language="python")

            # Code explanation
            st.info("**What this code does:**\n"
                   "- Applies the AI suggestions and your custom instructions\n"
                   "- Handles missing values, duplicates, and data type conversions\n"
                   "- Ensures data quality and consistency")

        # --- Execute Cleaning ---
        if "cleaning_code" in st.session_state:
            st.header("Step 6: Execute Cleaning")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Ready to clean your data?** This will apply all the transformations safely.")
            with col2:
                execute_button = st.button("Execute Cleaning", type="primary", use_container_width=True)

            if execute_button:
                with st.spinner(" Executing cleaning transformations..."):
                    try:
                        cleaned_df = execute_cleaning_code(st.session_state['current_df'], st.session_state['cleaning_code'])
                        st.session_state['cleaned_df'] = cleaned_df
                        st.session_state['cleaning_applied'] = True

                        # Calculate improvements
                        original_missing = st.session_state['original_df'].isnull().sum().sum()
                        cleaned_missing = cleaned_df.isnull().sum().sum()
                        missing_reduction = original_missing - cleaned_missing

                        st.success("Data cleaning completed successfully!")

                        # Show results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Rows", f"{st.session_state['original_df'].shape[0]:,}")
                        with col2:
                            st.metric("Cleaned Rows", f"{cleaned_df.shape[0]:,}")
                        with col3:
                            st.metric("Missing Values Fixed", missing_reduction)

                    except Exception as e:
                        st.error(f"Cleaning failed: {e}")
                        st.info("Try adjusting your instructions or check the generated code.")

            # Show cleaned data dashboard and comparison
            if "cleaned_df" in st.session_state:
                cleaned_df = st.session_state['cleaned_df']
                original_df = st.session_state['original_df']

                original_missing = original_df.isnull().sum().sum()
                cleaned_missing = cleaned_df.isnull().sum().sum()
                missing_reduction = original_missing - cleaned_missing
                duplicates_before = original_df.duplicated().sum()
                duplicates_after = cleaned_df.duplicated().sum()
                duplicate_reduction = duplicates_before - duplicates_after
                row_change = cleaned_df.shape[0] - original_df.shape[0]
                missing_pct = cleaned_missing / max(1, cleaned_df.size) * 100
                duplicate_pct = duplicates_after / max(1, cleaned_df.shape[0]) * 100
                quality_score = max(0, min(100, 100 - missing_pct - duplicate_pct))

                st.subheader("Cleaned Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cleaned Rows", f"{cleaned_df.shape[0]:,}", delta=f"{row_change:+,}")
                with col2:
                    st.metric("Missing Values", f"{cleaned_missing:,}", delta=f"-{missing_reduction:,}")
                with col3:
                    st.metric("Duplicate Rows", f"{duplicates_after:,}", delta=f"-{duplicate_reduction:,}")
                with col4:
                    st.metric("Data Quality Score", f"{quality_score:.1f}/100")

                st.markdown("---")
                st.subheader("Cleaned Data Comparison")
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.markdown("**Original Dataset**")
                    st.dataframe(original_df.head(8), use_container_width=True)
                with comp_col2:
                    st.markdown("**Cleaned Dataset**")
                    st.dataframe(cleaned_df.head(8), use_container_width=True)

                st.markdown("---")
                st.subheader("Data Cleaning Improvements")
                st.markdown(f"- **Missing values reduced by:** {missing_reduction:,}")
                st.markdown(f"- **Duplicate rows removed:** {duplicate_reduction:,}")
                st.markdown(f"- **Row count change:** {row_change:+,}")
                st.markdown(f"- **Data Quality Score:** {quality_score:.1f}/100")

                # --- Save & Export ---
                st.header("Step 7: Save & Export")
                version_id = f"v{len(get_history()) + 1}"
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Save to History", use_container_width=True):
                        save_cleaning_session(
                            version_id=version_id,
                            df=cleaned_df,
                            code=st.session_state["cleaning_code"],
                            instructions=final_instructions
                        )
                        st.success(f"Saved as {version_id}")

                with col2:
                    csv_data = cleaned_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name=f"{version_id}_cleaned.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col3:
                    st.download_button(
                        "Download Code",
                        data=st.session_state['cleaning_code'],
                        file_name=f"{version_id}_cleaning_code.py",
                        mime="text/x-python",
                        use_container_width=True
                    )

    else:
        # Show welcome message when no file is uploaded
        st.info(" **Get started:** Upload a CSV or Excel file to begin the cleaning process!")

        # Show sample workflow
        st.subheader(" How the Cleaning Process Works:")
        workflow_steps = [
            "**Upload** your raw dataset",
            "**Preview** data structure and quality issues",
            "**AI Analysis** generates smart cleaning suggestions",
            "**Customize** instructions based on your needs",
            "**Generate** optimized Python cleaning code",
            "**Execute** transformations safely",
            "**Save** results and download cleaned data"
        ]

        for step in workflow_steps:
            st.markdown(f"• {step}")

def show_data_analysis_page():
    """Data Analysis Page - Analyze cleaned datasets with ML and visualizations"""
    st.title("Advanced Data Analysis Dashboard")
    st.markdown("Perform statistical analysis, machine learning, and generate insights from your cleaned data.")

    # Check if cleaned data exists
    if "cleaned_df" not in st.session_state:
        st.warning("**No cleaned data available!**")
        st.info("Please go to the **Data Cleaning** page first to upload and clean your data.")
        return

    df = st.session_state['cleaned_df']
    analyzer = DataAnalyzer(df)

    # Dashboard Overview
    total_missing = df.isnull().sum().sum()
    total_duplicates = df.duplicated().sum()
    missing_pct = total_missing / max(1, df.size) * 100
    duplicate_pct = total_duplicates / max(1, df.shape[0]) * 100
    quality_score = max(0, min(100, 100 - missing_pct - duplicate_pct))

    st.header("Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns([1.2,1,1,1,1])
    with col1:
        st.markdown("**Dataset Metrics**")
        st.metric("Total Rows", f"{df.shape[0]:,}")
        st.metric("Total Columns", df.shape[1])
    with col2:
        st.markdown("**Quality**")
        st.metric("Missing Values", f"{total_missing:,}")
        st.metric("Missing %", f"{missing_pct:.2f}%")
    with col3:
        st.markdown("**Duplicates**")
        st.metric("Duplicate Rows", f"{total_duplicates:,}")
        st.metric("Duplicate %", f"{duplicate_pct:.2f}%")
    with col4:
        st.markdown("**Columns**")
        st.metric("Numeric", len(analyzer.numeric_cols))
        st.metric("Categorical", len(analyzer.categorical_cols))
    with col5:
        st.markdown("**Scorecard**")
        st.metric("Quality Score", f"{quality_score:.1f}/100")
        if quality_score >= 85:
            st.success("High quality")
        elif quality_score >= 65:
            st.info("Moderate quality")
        else:
            st.warning("Needs improvement")

    # Analysis Type Selection
    st.header(" Choose Analysis Type")
    analysis_type = st.selectbox(
        "Select analysis to perform:",
        ["Basic Statistics", "Correlation Analysis", "Outlier Detection",
         "Regression Analysis", "Clustering Analysis", "AI Insights", "Data Quality Report"]
    )

    # Basic Statistics
    if analysis_type == "Basic Statistics":
        st.subheader("Comprehensive Statistics")

        tab1, tab2, tab3 = st.tabs(["Numeric Summary", "Data Types", "Missing Values"])

        with tab1:
            if analyzer.numeric_cols:
                st.dataframe(df[analyzer.numeric_cols].describe(), use_container_width=True)

                # Distribution plots
                st.subheader("Distribution Visualizations")
                selected_col = st.selectbox("Select column to visualize:", analyzer.numeric_cols)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                df[selected_col].hist(bins=30, ax=ax1, alpha=0.7)
                ax1.set_title(f'Histogram of {selected_col}')
                ax1.set_xlabel(selected_col)
                ax1.set_ylabel('Frequency')

                df[selected_col].plot.box(ax=ax2)
                ax2.set_title(f'Box Plot of {selected_col}')

                st.pyplot(fig)
            else:
                st.info("No numeric columns found for statistical analysis.")

        with tab2:
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)

        with tab3:
            missing_summary = df.isnull().sum()
            if missing_summary.sum() > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_summary.index,
                    'Missing Count': missing_summary.values,
                    'Missing %': (missing_summary.values / len(df) * 100).round(2)
                }).sort_values('Missing Count', ascending=False)

                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)

                # Missing values heatmap
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df.isnull(), cbar=False, ax=ax)
                ax.set_title('Missing Values Heatmap')
                st.pyplot(fig)
            else:
                st.success("No missing values found in the dataset!")

    # Correlation Analysis
    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")

        if len(analyzer.numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
        else:
            corr_matrix = analyzer.correlation_analysis()

            # Correlation matrix
            st.dataframe(corr_matrix, use_container_width=True)

            # Heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax)
            ax.set_title('Correlation Heatmap (Upper Triangle)')
            st.pyplot(fig)

            # Strong correlations
            st.subheader("Strong Correlations (|r| > 0.7)")
            strong_corr = corr_matrix.where(np.abs(corr_matrix) > 0.7)
            strong_corr = strong_corr.stack().reset_index()
            strong_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']
            strong_corr = strong_corr[strong_corr['Variable 1'] != strong_corr['Variable 2']]
            strong_corr = strong_corr.drop_duplicates(subset=['Correlation'])

            if not strong_corr.empty:
                st.dataframe(strong_corr, use_container_width=True)
            else:
                st.info("No strong correlations found.")

    # Outlier Detection
    elif analysis_type == "Outlier Detection":
        st.subheader("Outlier Detection")

        method = st.selectbox("Detection Method:", ["IQR (Interquartile Range)", "Z-Score"])

        if st.button("Detect Outliers"):
            outliers = analyzer.outlier_detection('iqr' if method == "IQR (Interquartile Range)" else 'zscore')

            total_outliers = sum(len(indices) for indices in outliers.values())

            if total_outliers > 0:
                st.warning(f"Found {total_outliers} potential outliers across {sum(1 for v in outliers.values() if v)} columns.")

                # Summary table
                outlier_summary = pd.DataFrame({
                    'Column': list(outliers.keys()),
                    'Outlier Count': [len(indices) for indices in outliers.values()],
                    'Outlier %': [round(len(indices) / len(df) * 100, 2) for indices in outliers.values()]
                })
                st.dataframe(outlier_summary[outlier_summary['Outlier Count'] > 0], use_container_width=True)

                # Detailed view
                for col, indices in outliers.items():
                    if indices:
                        with st.expander(f"View {col} outliers ({len(indices)} found)"):
                            outlier_data = df.loc[indices]
                            st.dataframe(outlier_data, use_container_width=True)

                            # Visualization
                            fig, ax = plt.subplots(figsize=(8, 4))
                            df[col].hist(bins=30, ax=ax, alpha=0.7, label='All data')
                            ax.scatter(outlier_data[col], [0] * len(outlier_data), color='red', s=50, label='Outliers')
                            ax.set_title(f'{col} Distribution with Outliers')
                            ax.legend()
                            st.pyplot(fig)
            else:
                st.success("No outliers detected using the selected method!")

    # Regression Analysis
    elif analysis_type == "Regression Analysis":
        st.subheader("Linear Regression Analysis")

        if len(analyzer.numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for regression analysis.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                target = st.selectbox(
                    "Target Variable (Y):",
                    analyzer.numeric_cols,
                    key="regression_target"
                )

            with col2:
                features = st.multiselect(
                    "Feature Variables (X):",
                    [col for col in analyzer.numeric_cols if col != target],
                    default=analyzer.numeric_cols[:1] if len(analyzer.numeric_cols) > 1 else [],
                    key="regression_features"
                )

            if features and st.button("Run Regression Analysis"):
                try:
                    results = analyzer.perform_regression(target, features)

                    # Results display
                    st.success("Regression analysis completed!")

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{results['r2']:.3f}")
                    with col2:
                        st.metric("MSE", f"{results['mse']:.3f}")
                    with col3:
                        st.metric("RMSE", f"{results['mse']**0.5:.3f}")

                    # Coefficients
                    st.subheader("Model Coefficients")
                    coef_df = pd.DataFrame({
                        'Feature': features,
                        'Coefficient': results['coefficients']
                    })
                    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
                    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
                    coef_df = coef_df.drop('Abs_Coefficient', axis=1)
                    st.dataframe(coef_df, use_container_width=True)

                    st.info(f"**Intercept:** {results['intercept']:.3f}")

                    # Visualization
                    st.subheader("Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(results['predictions'], df[target].iloc[:len(results['predictions'])],
                             alpha=0.6, label='Data points')
                    ax.plot([min(results['predictions']), max(results['predictions'])],
                           [min(results['predictions']), max(results['predictions'])],
                           'r--', label='Perfect prediction')
                    ax.set_xlabel('Predicted Values')
                    ax.set_ylabel('Actual Values')
                    ax.set_title('Actual vs Predicted Values')
                    ax.legend()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Regression analysis failed: {e}")

    # Clustering Analysis
    elif analysis_type == "Clustering Analysis":
        st.subheader("K-Means Clustering Analysis")

        if len(analyzer.numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for clustering analysis.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                features = st.multiselect(
                    "Features for Clustering:",
                    analyzer.numeric_cols,
                    default=analyzer.numeric_cols[:2]
                )

            with col2:
                n_clusters = st.slider("Number of Clusters:", 2, 10, 3)

            if features and st.button("Run Clustering"):
                try:
                    results = analyzer.perform_clustering(features, n_clusters)

                    st.success("Clustering analysis completed!")

                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
                    with col2:
                        st.metric("Inertia", f"{results['inertia']:.3f}")

                    # Cluster distribution
                    st.subheader("Cluster Distribution")
                    cluster_counts = pd.Series(results['cluster_labels']).value_counts().sort_index()
                    cluster_df = pd.DataFrame({
                        'Cluster': cluster_counts.index,
                        'Count': cluster_counts.values,
                        'Percentage': (cluster_counts.values / len(df) * 100).round(1)
                    })
                    st.dataframe(cluster_df, use_container_width=True)

                    # Add cluster labels to data preview
                    df_with_clusters = df.copy()
                    df_with_clusters['Cluster'] = results['cluster_labels']
                    st.subheader("Data with Cluster Labels")
                    st.dataframe(df_with_clusters.head(20), use_container_width=True)

                    # Visualization (if 2 features selected)
                    if len(features) >= 2:
                        st.subheader("Cluster Visualization")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(df[features[0]], df[features[1]],
                                           c=results['cluster_labels'], cmap='viridis', alpha=0.6)
                        ax.set_xlabel(features[0])
                        ax.set_ylabel(features[1])
                        ax.set_title(f'K-means Clustering (k={n_clusters})')
                        plt.colorbar(scatter, ax=ax, label='Cluster')
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Clustering analysis failed: {e}")

    # AI Insights
    elif analysis_type == "AI Insights":
        st.subheader("AI-Powered Insights & Recommendations")

        if st.button("Generate AI Insights", type="primary"):
            with st.spinner("Analyzing your data and generating insights..."):
                # Gather comprehensive analysis results
                analysis_summary = {
                    "dataset_info": {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.astype(str).to_dict()
                    },
                    "basic_stats": analyzer.get_basic_stats(),
                    "correlations": analyzer.correlation_analysis().to_dict() if not analyzer.correlation_analysis().empty else {},
                    "outliers": analyzer.outlier_detection(),
                }

                insights = analyzer.generate_insights(analysis_summary)

            st.success("AI insights generated!")

            # Display insights in a nice format
            st.markdown("### Key Insights & Recommendations")
            st.markdown(insights)

            # Download insights
            st.download_button(
                "Download Insights Report",
                data=insights,
                file_name="data_analysis_insights.txt",
                mime="text/plain"
            )

    # Data Quality Report
    elif analysis_type == "Data Quality Report":
        st.subheader("Comprehensive Data Quality Report")

        # Generate comprehensive report
        report_data = {
            "Dataset Overview": {
                "Total Rows": df.shape[0],
                "Total Columns": df.shape[1],
                "Numeric Columns": len(analyzer.numeric_cols),
                "Categorical Columns": len(analyzer.categorical_cols),
                "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            },
            "Data Quality Metrics": {
                "Total Missing Values": df.isnull().sum().sum(),
                "Missing Value %": f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%",
                "Duplicate Rows": df.duplicated().sum(),
                "Duplicate %": f"{(df.duplicated().sum() / df.shape[0] * 100):.2f}%"
            }
        }

        # Display report
        for section, metrics in report_data.items():
            st.markdown(f"### {section}")
            for metric, value in metrics.items():
                st.metric(metric, value)

        # Column-wise quality
        st.subheader("Column Quality Analysis")
        quality_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': df.nunique(),
            'Unique %': (df.nunique() / len(df) * 100).round(2)
        })
        st.dataframe(quality_df, use_container_width=True)

        # Quality score
        missing_score = max(0, 100 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100))
        duplicate_score = max(0, 100 - (df.duplicated().sum() / df.shape[0] * 100))
        overall_score = (missing_score + duplicate_score) / 2

        st.subheader("Data Quality Score")
        st.metric("Overall Quality Score", f"{overall_score:.1f}/100")

        if overall_score >= 90:
            st.success(" Excellent data quality!")
        elif overall_score >= 70:
            st.info(" Good data quality with minor issues.")
        else:
            st.warning("Data quality needs improvement.")

if page == "Data Cleaning":
    show_data_cleaning_page()
elif page == "Data Analysis":
    show_data_analysis_page()

