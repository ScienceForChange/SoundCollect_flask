# build a docker image and run a container from it:
# docker build -t flask-server:0.5 .
# docker run -ti --rm -p 80:5000 -v ./:/flask-simple-app flask-server:0.5
# or this one? without --rm  docker run -ti -p 80:5000 -v .:/flask-simple-app flask-server:0.5


FROM python:3.10.0-slim-buster

WORKDIR /flask-simple-app

COPY . .

RUN pip3 install -r requirements.txt
RUN pip install flask
RUN pip install scikit-maad

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0" , "--debug"]
