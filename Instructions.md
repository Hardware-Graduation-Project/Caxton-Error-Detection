# Project Execution Guide

## Environment Setup

Navigate to the `\CAXTON-ERROR-DETECTION` directory and execute the following:

1. Create a virtual environment:  
   `python -m venv env`
2. Activate the virtual environment:  
   `env\Scripts\activate`
3. Install required dependencies:  
   `pip install -r requirements.txt`

## Post-Training Execution

In the `\src` directory:

1. Copy the trained model's `lightning_logs` directory here
2. Copy the trained model's `checkpoints` directory here
3. Configure and run the model with these commands:
   1. Set data directory path:  
      `$env:DATA_DIR ="..\data"`
   2. Specify checkpoint path:  
      `$env:CHECKPOINT_PATH = ".\lightning_logs\version_1\checkpoints\epoch=0-step=1217.ckpt"`
   3. Execute sample processing:  
      `python samples.py`  
      (This processes examples in `\data\cropped`)
