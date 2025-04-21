# Use the official Python image as base
FROM python:3.11-bullseye

# Install build-essential for gcc and other build tools

# Install base build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    gpg \
    software-properties-common \
    ca-certificates \
    && apt-get clean

# Install libstdc++ from newer GCC (to get GLIBCXX_3.4.32)
RUN wget -qO - https://ftp.gnu.org/gnu/gcc/gcc-13.2.0/gcc-13.2.0.tar.gz | tar xz \
    && cd gcc-13.2.0 \
    && ./contrib/download_prerequisites \
    && mkdir build && cd build \
    && ../configure --disable-multilib --enable-languages=c,c++ \
    && make -j$(nproc) \
    && make install

# Verify the correct version is present
RUN strings /usr/local/lib64/libstdc++.so.6 | grep GLIBCXX

# Set environment variables to use new compiler
ENV LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
ENV CC=/usr/local/bin/gcc
ENV CXX=/usr/local/bin/g++


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
