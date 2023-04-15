FROM python:3.9.16
EXPOSE 8501
WORKDIR /src/src
COPY requirements_3916_docker.txt ./requirements_3916_docker.txt
COPY src/haarcascade_frontalface_default.xml ./src/src/haarcascade_frontalface_default.xml
COPY src/final_model.h5 ./src/src/final_model.h5
RUN pip3 install -r requirements_stripped.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY . .
# CMD ls
# CMD pwd
CMD streamlit run src/app.py \
    --server.headless true \
    --browser.serverAddress="0.0.0.0" \
    --server.enableCORS false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats false