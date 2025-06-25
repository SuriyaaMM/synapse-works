# Synapse Works

## Overview
Architect, Train & Evaluate your Neural Networks in seconds

### Features
- Intuitive UI for creating neural networks, training them and visualizing them
- Supports multiple backends (although currently only pytorch)(future implementations might have keras, tf and jax)
- Visualize `gradient_norm`, `loss`, `accuracy`, `hyperparameters`, `flow of gradients`, `weight & biases`, `computation graph`

### Gallery
[Visit Gallery](./gallery/README.md)

### Instructions for Building DockerImage
```bash
sudo docker build -t synapse-works .
```
```bash
sudo docker run -it --rm   -p 5173:5173 -p 4000:4000 -p 6379:6379 -p 6000:6000  synapse-works
```

## Installation
### Pre-requsities
- [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)
- [redis-server](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/)
- Note: you have to start redis-server using `sudo systemctl enable redis-server`

### Clone the repo
```bash
git clone https://github.com/SuriyaaMM/synapse-works
cd synapse_works
```
### Create `conda` environment (Recommended)
```bash
conda create env -n synapse-works python=3.12
conda update env -f environment.yaml
```
### Initialize npm modules
```bash
npm install && cd frontend && npm install && cd ../
```

## Instructions for Running
Have three terminals open, in first one
```bash
npm start
```
in second one
```bash
python service/worker.py
```
in third one
```bash
cd frontend
npm run dev
```
