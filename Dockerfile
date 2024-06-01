FROM python:3.11-slim

ENV POETRY_VERSION=1.8.2

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install poetry==${POETRY_VERSION} \
  && poetry config virtualenvs.create false \
  && poetry install --no-root --no-interaction --only main

COPY src ./

EXPOSE 5000

CMD ["python", "-m", "model_service.app"]
