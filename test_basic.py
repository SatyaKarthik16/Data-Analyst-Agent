"""
Basic tests for the Data Analyst Agent application.
"""

import pandas as pd
import pytest
from analyzer import analyze_dataframe
from data_analysis import DataAnalyzer

def test_analyzer():
    """Test data analyzer functionality."""
    df = pd.DataFrame({
        'a': [1, 2, None],
        'b': [3, 4, 5],
        'c': ['x', 'y', 'z']
    })
    summary = analyze_dataframe(df)
    assert '3 rows' in summary
    assert '3 columns' in summary

def test_data_analyzer():
    """Test data analysis functionality."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10]
    })
    analyzer = DataAnalyzer(df)
    stats = analyzer.get_basic_stats()
    assert stats['shape'] == (5, 2)
    assert len(stats['numeric_summary']) == 2

if __name__ == "__main__":
    test_analyzer()
    test_data_analyzer()
    print("All tests passed!")