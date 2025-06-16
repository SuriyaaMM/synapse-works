# Synapse Works

## Overview
Architect, Train & Evaluate your Neural Networks in seconds

## Installation
### Pre-requsities
- [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)
- [redis-server](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/)
### Clone the repo
```bash
git clone https://github.com/SuriyaaMM/synapse-works
cd synapse_works
```
### Create `conda` environment (Recommended)
```bash
conda create env -f environment.yaml
```
### Initialize npm modules
```bash
npm install
```

## Instructions for Running
Have two terminals open, in first one
```bash
npm start
```
in second one
```bash
python service/worker.py
```
