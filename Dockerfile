# build a docker image and run a container from it:
# docker build -t flask-server:0.5 .
# docker run -ti --rm -p 80:5000 -v ./:/flask-simple-app flask-server:0.5
# or docker compose up??

FROM python:3.10.0-slim-buster

WORKDIR /flask-simple-app

COPY ./requirements.txt .

RUN pip3 install -r requirements.txt

# RUN pip install flask
RUN pip install scikit-maad
RUN pip install -U flask-cors
RUN apt update
RUN apt install git -y
RUN pip install git+https://github.com/endolith/waveform_analysis.git@master

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0" , "--debug"]
