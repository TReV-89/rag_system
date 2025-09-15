FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY streamlit_ui.py . 
COPY rag_methods.py .


EXPOSE 8501

CMD ["streamlit", "run", "streamlit_ui.py"]
