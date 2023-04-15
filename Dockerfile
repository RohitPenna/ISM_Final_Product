FROM python:3.9.16
EXPOSE 8501
WORKDIR /src
COPY requirements_stripped.txt ./requirements_stripped.txt
RUN pip3 install -r requirements_stripped.txt
COPY . .
CMD streamlit run app.py \
    --server.headless true \
    --browser.serverAddress="0.0.0.0" \
    --server.enableCORS false \
    --browser.gatherUsageStats false