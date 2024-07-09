FROM python:3.12
WORKDIR /movie_pred
COPY /requirements.txt /movie_pred/requirements.txt
RUN pip install -r /movie_pred/requirements.txt
COPY /app /movie_pred/app
COPY /utils /movie_pred/utils

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]