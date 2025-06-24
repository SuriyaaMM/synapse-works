# nodejs bookworm image
FROM node:24-bookworm-slim

# Set the working directory in the container
WORKDIR /app

# --- Install system dependencies for Conda and build tools ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    build-essential \ 
    libssl-dev \ 
    libffi-dev \ 
    && rm -rf /var/lib/apt/lists/*

# --- Install Miniconda ---
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Initialize conda
RUN conda init bash

# --- Install Python 3.12 via Conda ---
RUN conda install -y python=3.12

# Create a symlink for consistency (optional)
RUN ln -sf /opt/conda/bin/python /usr/local/bin/python

# Verify Python installation
RUN python --version
RUN pip --version

# --- Install Redis Server --- 
RUN apt-get update && apt-get install -y --no-install-recommends redis-server \
    && rm -rf /var/lib/apt/lists/*

# Verify Redis version
RUN redis-server --version

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Expose ports for your Python app and Redis
EXPOSE 4000
EXPOSE 6000
EXPOSE 5173
EXPOSE 6379

# Install node modules for frontend & backend
RUN chmod +x install.sh

# Command to run when the container launches
CMD ["./start.sh"]