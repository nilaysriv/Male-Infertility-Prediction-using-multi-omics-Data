# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies (if any are needed for your Python packages)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY ./app /usr/src/app/app

# Expose the port the app runs on
EXPOSE 8000

# Specify the command to run on container start
# This command runs the Gunicorn server, binding it to all available network interfaces on port 8000.
# It assumes your Flask app instance is named 'app' inside 'app/main.py'.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app.main:app"]
