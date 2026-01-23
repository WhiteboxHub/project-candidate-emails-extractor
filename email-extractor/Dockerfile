# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the project code
COPY . .

# Default environment variables
ENV ENV_MODE=test
# ENV ENV_MODE=service
# Entrypoint script to choose between test or service
ENTRYPOINT ["sh", "-c"]
CMD ["if [ \"$ENV_MODE\" = 'test' ]; then python test_my_account.py; else python service.py; fi"]
