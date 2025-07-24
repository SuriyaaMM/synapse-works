# Synapse Works

## Overview

Architect, Train & Evaluate your Neural Networks in seconds, without writing any code. Synapse Works provides an intuitive, web-based visual framework for rapid neural network development.

### Features
- **Visual Builder:** An intuitive node-based UI for creating, training, and visualizing neural networks.
- **Backend Support:** Built on PyTorch, with a modular design for future expansion to TensorFlow, Keras, and JAX.
- **Rich Visualization:** Monitor `gradient_norm`, `loss`, `accuracy`, training `hyperparameters`, `gradient flow`, `weight & bias` distributions, and the `computation graph`.

### Gallery
[**Visit the Gallery**](./gallery/README.md) to see what you can build!

---
## Quick Start (Docker)

This is the fastest and recommended way to get started.

Pull the Pre-built Image
```bash
sudo docker pull suriyaamm2705/synapse-works:v1.4.0
```

Run the Container

```bash
sudo docker run -it --rm   -p 5173:5173 -p 4000:4000 -p 6379:6379 -p 6000:6000  synapse-works
```

Access the Application
Once the container is running, open your browser and go to **`http://localhost:5173`**.

-----

### Install Python Dependencies (Choose One Path)

You can install the required Python packages using either Conda (recommended for a self-contained environment) or Pip (if you have an existing Python 3.12 setup).

#### **Path 1: Using Conda (Recommended)**

This method creates a completely new, isolated Conda environment with all dependencies managed automatically.

```bash
conda create -n synapse-works python=3.12.11
conda activate synapse-works
conda env update --file environment.yaml
conda activate synapse-works
```

-----

#### **Path 2: Using Pip and an Existing Python Environment**

Install packages from `requirements.txt`

```bash
pip install -r requirements.txt
```

**4. Install NPM Modules**
Install dependencies for both the root directory and the frontend.

```bash
# Install root dependencies
npm install

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running the Application

You need to run three separate processes in three different terminal tabs. Make sure the `synapse-works` conda environment is activated for the Python worker.

**Terminal 1: Start the Backend API**

```bash
# (In project root)
npm start
```

**Terminal 2: Start the Python Worker**

```bash
# (In project root, with conda env activated)
conda activate synapse-works
python service/worker.py
```

**Terminal 3: Start the Frontend**

```bash
# (In project root)
cd frontend
npm run dev
```

After all services are running, access the application at **`http://localhost:5173`**.
