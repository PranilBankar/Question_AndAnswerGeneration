# Use the official Python 3.10 image as base
FROM python:3.10-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire project into the container
# .dockerignore will prevent .venv and other large unnecessary files from being copied
COPY . /code

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Command to run the FastAPI application
CMD ["uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
