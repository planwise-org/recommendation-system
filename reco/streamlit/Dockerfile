# Use the official Python image as base
FROM python:3.11-bullseye

# Install build-essential for gcc and other build tools
# RUN apt-get update && apt-get install -y build-essential

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    gcc-11 \
    g++-11 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# RUN apt-get update && apt-get install -y \
#     build-essential \
    # gcc-11 \
    # g++-11 \
    # libstdc++6 \
    # wget \
    # curl \
    # && apt-get clean

# Set gcc to use the newer version (optional, helpful if compiling code)
# ENV CC=gcc-11
# ENV CXX=g++-11


# RUN apt-get update && apt-get install -y \
# build-essential \
# gcc-11 \
# g++-11 \
# libstdc++6 \
# && apt-get clean

RUN pip install --upgrade pip
RUN pip install Cython "numpy<2.0.0" "scipy<=1.13.1" tqdm powerlaw


# Set the working directory inside the container
WORKDIR /

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt


# Copy the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8501

# Command to run FastAPI with Uvicorn
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
