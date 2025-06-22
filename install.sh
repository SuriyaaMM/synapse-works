echo "creating conda environment"
conda create env -n synapse-works python=3.12 && conda update env -f environment.yaml

echo "activating conda environment"
conda activate synapse-works

echo "installing node modules"
npm install && cd frontend && npm install