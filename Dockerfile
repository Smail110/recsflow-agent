FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY recagent ./recagent
COPY app.py ./
COPY .streamlit ./.streamlit
EXPOSE 8501 8000
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]

