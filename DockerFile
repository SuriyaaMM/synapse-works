# nodejs bookworm image
FROM node:24-bookworm-slim

# Set the working directory in the container
WORKDIR /app

# --- Install Python 3.12 ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3.12-venv \ 
    build-essential \ 
    libssl-dev \ 
    libffi-dev \ 
    && rm -rf /var/lib/apt/lists/*

# Link python3 to python if not already done, for consistency
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Verify Python and pip installations
RUN python --version
RUN pip --version

# --- Install Redis Server --- 
RUN apt-get update && apt-get install -y --no-install-recommends redis-server \
    && rm -rf /var/lib/apt/lists/*

# Verify Redis version (it might not be exactly 7.0.15, but should be close)
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