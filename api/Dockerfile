# Use the official Python image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install  -r requirements.txt

# Copy the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8080

# Command to run FastAPI with Uvicorn
WORKDIR /src
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
