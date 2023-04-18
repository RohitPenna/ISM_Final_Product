# ISM_Final_Product

Interactive Chatbot with Real-Time Emotional Analysis/Integration

## Usage

Option 1 (Run locally):

- Clone this repository
- Enter the commands:

  - Make a virtual environment: `python -m venv .venv`
  - Activate it (for Windows command prompt): `.venv\Scripts\activate`
  - `pip install requirements.txt -r`
  - Open an OpenAI account at `https://platform.openai.com/account/org-settings`
  - Add a .env file to the top level directory with the following environment variables.

    ```
    OPENAI_API_KEY=xxxx
    OPENAI_ORG_NAME=xxxxx
    ```

  - `cd src/`
  - `streamlit run app.py`

Option 2 (Docker):

- Download Docker Desktop
- `docker pull rcsnyder/streamlitapp:0.0.1`
- Enter into the app: `sudo docker run --rm -it -v /$(pwd)/src -p 8501:8501 --env OPENAI_API_KEY=xxxx --env OPENAI_ORG_NAME=xxxx --entrypoint bash rcsnyder/streamlitapp:0.0.1`
- (Optional) Add a .env file to the top level directory with the following environment variables.

  ```
  OPENAI_API_KEY=xxxx
  OPENAI_ORG_NAME=xxxxx

  ```

- Start it: `streamlit run app.py --server.headless true --browser.serverAddress="0.0.0.0" --server.enableCORS false --server.enableXsrfProtection=false --browser.gatherUsageStats false`
- Go to `localhost:8501`
- Choose camera device
- Click Start

Option 3 (Deploy on AWS EC2):

- Spin up an EC2 instance on AWS with Ubuntu 22.04. 16gb of RAM, 128gb EBS
- Set the security group for port 80 and 443 to all IP addresses.
- Set a pem key
- Windows Cmds:
  - `icacls.exe filename.pem /reset`
  - `icacls.exe filename.pem /grant:r %username%:(R)`
  - `icacls.exe filename.pem /inheritance:r`
  - `ssh -i "filename.pem" ubuntu@ec2instancepublicaddress`
- Linux Commands:
  - `chmod filename/pem 400`
  - `ssh -i "filename.pem" ubuntu@ec2instancepublicaddress`
- EC2 HTTPS encryption setup commands:
  - `sudo apt-get install -y debian-keyring`
  - `sudo apt-get install -y debian-archive-keyring`
  - `sudo apt-get install -y apt-transport-https`
  - `sudo keyring_location=/usr/share/keyrings/caddy-stable-archive-keyring.gpg`
  - `curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/setup.deb.sh' | sudo -E bash`
  - `sudo apt-get install caddy=2.6.4`
  - `sudo caddy start`
- EC2 App commands:

  - `sudo apt-get update`
  - `sudo apt-get remove docker docker-engine docker.io containerd runc`
  - `sudo apt-get install ca-certificates curl gnupg`
  - `curl -fsSL https://get.docker.com -o get-docker.sh`
  - `sudo sh get-docker.sh`
  - `sudo docker pull rcsnyder/streamlitapp:0.0.1`
  - `sudo docker run -it --rm -p 8501:8501 --env OPENAI_API_KEY=xxxx --env OPENAI_ORG_NAME=xxxx rcsnyder/streamlitapp:0.0.2`
  - (Optional to get into the docker container)
    - `sudo docker run --env OPENAI_API_KEY=xxxxx --env OPENAI_ORG_NAME=xxxx --rm -it -v /$(pwd)/src -p 8501:8501 --entrypoint bash rcsnyder/streamlitapp:0.0.1`
    - `streamlit run app.py --server.headless true --browser.serverAddress="0.0.0.0" --server.enableCORS false --server.enableXsrfProtection=false --browser.gatherUsageStats false`

- Go to `publicEC2IP.nip.io` and choose your camera device and click start.

### User Stories

- [ x ] A user would like to be able to start up an app and talk to a chatbot that can percieve their emotions in real-time.
- [ ] A user would like to be able to choose their initial meta prompt when talking to the chatbot that can percieve their emotions in real-time.

### Technology Stack

- Python
- Open CV for webcam image processing
- Streamlit or Gradio
- Huggingface Endpoints for emotion detecting pre-trained models
- OpenAI large language models (Chat GPT)
- Hypermodern Python Coding Standards (<https://cookiecutter-hypermodern-python.readthedocs.io/en/2021.3.14/guide.html>)

### Inspiration Repos/Articles

- <https://github.com/hwchase17/langchain>
- <https://github.com/omar178/Emotion-recognition>
- <https://github.com/gradio-app/hub-emotion-recognition>
- <https://github.com/joeychrys/streamlit-chatGPT/blob/master/src/main.py>
- <https://blog.streamlit.io/how-to-build-the-streamlit-webrtc-component/>
- <https://github.com/nicolalandro/yolov5_streamlit>
- <https://github.com/nextstep-infotech/Open-AI/blob/main/streamlit_chatGPT_clone.py>
- <https://blog.devgenius.io/facial-emotion-detection-using-aws-rekognition-in-python-69b2da668192>
- <https://github.com/codedev99/fast-face-exp>
- <https://github.com/sanjeevm4788/Face-emotion-detection>
- <https://stackoverflow.com/questions/69439390/streamlit-image-processing>
- <https://github.com/whitphx/streamlit-fesion>
- <https://github.com/whitphx/streamlit-webrtc/blob/56d715a76247cd6303c9ddade5bc2a8e430ddc2f/app.py#L344>
- <https://bansalanuj.com/https-aws-ec2-without-custom-domain>
- <https://github.com/nicolasmetallo/deploy-streamlit-on-fargate-with-aws-cdk>

### Huggingface

- <https://huggingface.co/spaces/schibsted/Facial_Recognition_with_Sentiment_Detector>
- <https://huggingface.co/spaces/schibsted/facial_expression_classifier/blob/main/app.py>

### Page Breakdown

- Main page
  - Continuous Webcam Viewport (Potentially with a start and stop button)
  - Text Input/Output textbox with a submit button

### Implementation

- User Data
  - Detected Emotion
  - Prompt/Prompt History
- User Interaction
  - Buttons for camera start and stop
  - Button to submit prompt
- API Calls
  - Emotion Detection Model to get detected emotion
  - Large Language Model to get response to do something with the detected emotion

# TODO

- [ ] Flip the conversation history to show most recent messages first
- [ ] Make the form input chat text clear on change.
- [x] figure out how to continuously grab the video image to pipe into another emotion detection api
- [ ] figure out how to continuously update state with the emotion dictionary and wire it up to the chatbot prompt with the average since last updated.
