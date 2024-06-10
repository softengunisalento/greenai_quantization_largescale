# 🥑 Green AI Quantization

## Requirements

### Virtual environment

Need to use `.env`

To be able to run a model we need to use a **virtual environment**, so create it in your project directory:

```bash
python3 -m venv .env
```

Activate the virtual environment. On Linux and MacOS:

```bash
source .env/bin/activate
```

To deactivate it. On Linux and MacOS:

```
deactivate
```

### Libraries

Can just run the script `requirements.sh` :

```bash
./requirements.sh
```

or manually install all the libraries

```bash
pip install transformers
pip install quanto
pip install accelerate
pip install datasets
pip install setuptools
pip install bitsandbytes
pip install torch
pip install torchvision
pip install torchaudio
```