# Setup Guide for DocuMind AI

Welcome to the DocuMind AI setup guide. This document contains instructions for configuring and running the application.

## Prerequisites

Before running the application, make sure you have the following packages installed:

- Python 3.11 or higher
- Docker and Docker Compose (optional, for containerized run)
- Git (for cloning the repository)

## Local Installation

To install the application locally, follow these steps:

1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/username/documind-ai.git
   cd documind-ai
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```

Open the `.env` file and configure your `OPENAI_API_KEY`:
```env
OPENAI_API_KEY=sk-yourKeyHere
```

## Running the Application

Start the Streamlit interface using the following command:
```bash
streamlit run app.py
```
By default, the application will be hosted at `http://localhost:8501`.
