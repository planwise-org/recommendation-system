# Use the official Python image as base
FROM python:3.11

# Install build-essential for gcc and other build tools
# RUN apt-get update && apt-get install -y build-essential

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*



RUN pip install --upgrade pip
RUN pip install Cython "numpy<2.0.0" "scipy<=1.13.1" tqdm powerlaw


# Set the working directory inside the container
WORKDIR /

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the application code
# COPY . .

# Expose the FastAPI port
# EXPOSE 8501

# Command to run FastAPI with Uvicorn
# CMD ["streamlit", "run", "app.py"]
