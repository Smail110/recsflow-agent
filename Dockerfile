FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt
# src-layout: the package lives under src/ and is importable from the workdir.
COPY src ./src
COPY docs/contract ./docs/contract
COPY app.py ./
COPY .streamlit ./.streamlit
ENV PYTHONPATH=/app/src
RUN useradd --create-home --uid 10001 recagent
ENV HOME=/home/recagent
USER recagent
EXPOSE 8501 8000
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
