FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "sqlalchemy>=2.0,<2.1" \
    "alembic>=1.13,<1.14" \
    mlflow==2.16.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.35.36

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
