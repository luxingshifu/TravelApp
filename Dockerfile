from python:3

MAINTAINER williamcottrell72@gmail.com

COPY . /app
WORKDIR /app

RUN pip install pipenv

RUN pipenv install --system --deploy 

EXPOSE 80

EXPOSE 5432
ENTRYPOINT["python"]
CMD ["python", "app.py"]
