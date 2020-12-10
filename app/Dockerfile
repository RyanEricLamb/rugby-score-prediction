FROM python:3.7.3-stretch

RUN mkdir /app
WORKDIR /app

#Copy all files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run
CMD ["python","./app.py"]


