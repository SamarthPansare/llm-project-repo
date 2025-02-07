FROM python:3.9-slim

# set working dir as app
WORKDIR /app

# install dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    curl \
    poppler-utils \   
    && rm -rf /var/lib/apt/lists/*

# install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh  
# Verify Ollama installation
RUN ollama --version

RUN ollama serve & sleep 5 && ollama pull deepseek-r1:1.5b

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy files of this dir in container
COPY . .

RUN echo "Starting Python Application"
# run 
ENTRYPOINT ["sh", "-c", "ollama serve & sleep 5 && python main.py"]

