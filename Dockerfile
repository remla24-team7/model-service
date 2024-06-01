FROM tensorflow/tensorflow:2.16.1

ENV POETRY_VERSION=1.8.2

RUN pip install poetry==${POETRY_VERSION}
RUN poetry config virtualenvs.options.system-site-packages true

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --only main

COPY src ./

EXPOSE 5000

CMD ["poetry", "run", "python", "-m", "model_service.app"]
