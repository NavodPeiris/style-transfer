FROM python:3.10-slim
WORKDIR /src
COPY ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./src ./
CMD ["uvicorn", "test:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
