"""
Data analysis module for the Data Analyst application.

Provides exploratory data analysis, statistical tests, and ML model analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import OPENAI_API_KEY, MODEL_NAME, logger

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Performs various data analysis tasks."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(include='object').columns.tolist()

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistical summary."""
        stats = {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes.to_dict(),
            'missing': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum()
        }

        if self.numeric_cols:
            stats['numeric_summary'] = self.df[self.numeric_cols].describe().to_dict()

        return stats

    def correlation_analysis(self) -> pd.DataFrame:
        """Compute correlation matrix for numeric columns."""
        if len(self.numeric_cols) < 2:
            return pd.DataFrame()
        return self.df[self.numeric_cols].corr()

    def outlier_detection(self, method: str = 'iqr') -> Dict[str, List[int]]:
        """Detect outliers using IQR or Z-score method."""
        outliers = {}
        for col in self.numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].index.tolist()
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers[col] = self.df[z_scores > 3].index.tolist()
        return outliers

    def perform_regression(self, target: str, features: List[str]) -> Dict[str, Any]:
        """Perform linear regression analysis."""
        if target not in self.df.columns or not all(f in self.df.columns for f in features):
            raise ValueError("Invalid target or features")

        X = self.df[features].fillna(self.df[features].mean())
        y = self.df[target].fillna(self.df[target].mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results = {
            'coefficients': dict(zip(features, model.coef_)),
            'intercept': model.intercept_,
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'predictions': y_pred[:10].tolist()  # Sample predictions
        }

        logger.info(f"Regression completed for {target}")
        return results

    def perform_clustering(self, features: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """Perform K-means clustering."""
        if not all(f in self.df.columns for f in features):
            raise ValueError("Invalid features")

        X = self.df[features].fillna(self.df[features].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        results = {
            'cluster_labels': clusters.tolist(),
            'centroids': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_score(X_scaled, clusters)
        }

        logger.info(f"Clustering completed with {n_clusters} clusters")
        return results

    def generate_insights(self, analysis_results: Dict[str, Any]) -> str:
        """Generate AI-powered insights from analysis."""
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=MODEL_NAME)

        prompt = f"""
        Based on the following data analysis results, provide key insights and recommendations:

        {analysis_results}

        Focus on actionable insights for data-driven decision making.
        """

        messages = [
            SystemMessage(content="You are a data analysis expert. Provide clear, concise insights."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    def create_visualizations(self, output_dir: str = "visualizations") -> List[str]:
        """Generate basic visualizations."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        plots = []

        # Correlation heatmap
        if len(self.numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.df[self.numeric_cols].corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.savefig(f"{output_dir}/correlation_heatmap.png")
            plt.close()
            plots.append(f"{output_dir}/correlation_heatmap.png")

        # Distribution plots for numeric columns
        for col in self.numeric_cols[:5]:  # Limit to first 5
            plt.figure(figsize=(8, 6))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(f"{output_dir}/{col}_distribution.png")
            plt.close()
            plots.append(f"{output_dir}/{col}_distribution.png")

        logger.info(f"Generated {len(plots)} visualizations")
        return plots