# Data Analyst Agent

A comprehensive AI-powered data analysis platform that delivers secure, repeatable data cleaning and analytics workflows for business teams.

## Project Overview

This application is built as a productized workflow for data analysts and analytics teams. It combines:

- dataset upload and preview
- AI-powered data quality diagnosis
- generated Python cleaning code
- secure execution of that code against the dataset
- cleaned vs original comparison and export
- analytics dashboards, regression, clustering, and AI insights

Unlike a standalone prompt to ChatGPT or Claude, this tool is a fully integrated solution that turns AI recommendations into a governed, repeatable process.

## Business Case

### Problem we solve

Data analysts often spend 50–80% of their time on data preparation, including:

- identifying missing values and duplicates
- choosing cleaning strategies
- writing and validating transformation code
- preserving audit trails and version history
- manually moving from cleaning to analysis

### What this application delivers

- a single, consistent workflow from raw data to cleaned data and analysis
- safe and repeatable application of AI-generated cleaning code
- built-in dataset quality scoring and comparison metrics
- exportable cleaned data and transformation code
- an analyst-friendly UI with dashboards and reporting

## Value Compared to ChatGPT / Claude

ChatGPT and Claude are strong AI assistants, but they are not a complete data cleaning product. This project adds value by providing:

- dataset-specific cleaning, not just general advice
- direct file upload + preview + problem detection
- secure code execution rather than trusting raw generated scripts
- side-by-side cleaned vs raw data comparison
- reusable history, versioning, and export capabilities
- a fully branded Streamlit interface for analyst adoption

## Business Impact

For a firm that hires data analysts, this application can:

- reduce repetitive manual cleanup work
- improve consistency and quality across projects
- accelerate the time from data ingestion to insight
- lower risk from analyst-generated errors
- make cleaning and analysis more auditable and repeatable
- scale a team’s work with a shared product instead of siloed prompt sessions

## Features

- **Data Extraction**: Load data from files, APIs, and databases
- **Data Cleaning**: AI-assisted cleaning with secure code execution
- **Data Analysis**: Statistical analysis, ML models, and AI insights
- **Agentic Workflow**: LangGraph orchestration with state memory
- **Professional UI**: Streamlit-based interface
- **CLI Support**: Command-line interface for automation

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`
4. Run the application: `streamlit run app.py`

## Usage

### Web Interface
```bash
streamlit run app.py
```

### Command Line
```bash
python main.py path/to/dataset.csv --task full
```

## Security

- Uses RestrictedPython for safe code execution
- Environment variable management
- Input validation and error handling

## Architecture

- `data_loader.py`: Data loading from multiple file formats
- `analyzer.py`: Comprehensive dataset analysis
- `llm_suggester.py`: AI-powered cleaning suggestions
- `code_generator.py`: LLM-based code generation
- `executor.py`: Secure code execution
- `data_analysis.py`: Statistical and ML analysis
- `agent.py`: LangGraph workflow orchestration
- `app.py`: Professional Streamlit interface

## Technologies

- LangChain & LangGraph for AI orchestration
- OpenAI GPT for intelligent analysis
- Scikit-learn for ML analysis
- Streamlit for professional UI
- RestrictedPython for secure execution

## Professional Enhancements

- Removed emojis for professional appearance
- Added comprehensive logging
- Secure code execution with RestrictedPython
- ML analysis capabilities (regression, clustering)
- AI-generated insights
- Modular, scalable architecture
- Type hints and docstrings throughout

##  Tech Stack

| Tool        | Purpose                     |
|-------------|-----------------------------|
| LangChain   | LLM orchestration            |
| LangGraph   | Agent workflow/state engine  |
| OpenAI GPT  | LLM for analysis + code gen  |
| Streamlit   | Web UI frontend              |
| Pandas      | Data handling                |
| Python-dotenv | API key management         |
| JSON        | Version history tracking     |

---

 Final Project Structure

![image](https://github.com/user-attachments/assets/d502533c-b946-484d-8182-9d6a2400616d)

