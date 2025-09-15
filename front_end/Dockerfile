FROM python:3.12-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY streamlit_ui.py .
COPY rag_methods.py .

# Expose port (Render will use the PORT environment variable)
EXPOSE 8501

# Updated CMD for Render compatibility
CMD ["streamlit", "run", "streamlit_ui.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
