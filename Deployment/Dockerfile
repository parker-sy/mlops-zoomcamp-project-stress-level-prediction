FROM python:3.12-slim

WORKDIR /app
COPY Data /app/Data/
COPY Workflow /app/Workflow/
COPY Models /app/Models/
COPY app.py /app/app.py
COPY Pipfile /app/Pipfile
COPY Pipfile.lock /app/Pipfile.lock

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install pipenv
RUN pipenv install
# Install dependencies using Pipfile.lock only
# RUN pipenv install --deploy --ignore-pipfile
RUN echo "Inference process finished sucessfully"

CMD ["pipenv", "run", "python", "/app/app.py"]