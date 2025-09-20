# LLMs From Scratch

This repository is dedicated to building and understanding Large Language Models (LLMs) from the ground up using PyTorch. It provides a clear and concise implementation of popular transformer-based architectures like BERT, GPT, and ALBERT. This project is intended for educational purposes, to help developers and researchers understand the inner workings of these complex models.

## Features

*   **Pure PyTorch Implementation:** The models are built using only PyTorch, making the code easy to follow and modify.
*   **Classic Architectures:** Includes implementations of several well-known transformer architectures.
*   **Modular Design:** The code is organized into modules for easy understanding and extension.
*   **Training and Inference Scripts:** Comes with scripts to train your own models and to use them for predictions.
*   **Minimal Dependencies:** The project has minimal dependencies, making it easy to set up and run.

## Architecture Overview
```
Transformer
├── Encoder (Stack of 6 EncoderBlocks)
│   ├── Multi-Head Self-Attention
│   ├── Feed-Forward Network  
│   ├── Residual Connections
│   └── Layer Normalization
│
├── Decoder (Stack of 6 DecoderBlocks)  
│   ├── Masked Multi-Head Self-Attention
│   ├── Multi-Head Cross-Attention
│   ├── Feed-Forward Network
│   ├── Residual Connections
│   └── Layer Normalization
│
├── Input Embedding + Positional Encoding
└── Output Linear Layer
```

## Project Structure

The repository is organized as follows:

```
llms-from-scratch/
├── archs/
│   ├── __init__.py
│   ├── albert.py
│   ├── bert.py
│   ├── gpt.py
│   ├── roberta.py
│   └── transformer.py
├── src/
│   ├── __init__.py
│   ├── eval.py
│   ├── predict.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   └── test_transformer.py
├── experiments.ipynb
├── README.md
└── requirements.txt
```

-   **`archs/`**: Contains the implementation of different transformer architectures.
-   **`src/`**: Includes scripts for training (`train.py`), evaluation (`eval.py`), and prediction (`predict.py`).
-   **`tests/`**: Contains unit tests for the implemented models.
-   **`experiments.ipynb`**: A Jupyter notebook for experimenting with the models.
-   **`requirements.txt`**: A list of dependencies for the project.

## Getting Started

### Prerequisites

-   Python 3.7+
-   PyTorch

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thedatamonk/llms-from-scratch.git
    cd llms-from-scratch
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train a model, you can use the `train.py` script. You will need to provide a dataset and specify the model architecture you want to use.

```bash
python src/train.py --model <model_name> --dataset <path_to_dataset>
```

### Evaluation

To evaluate a trained model, use the `eval.py` script.

```bash
python src/eval.py --model_path <path_to_model> --dataset <path_to_dataset>
```

### Prediction

To make predictions with a trained model, use the `predict.py` script.

```bash
python src/predict.py --model_path <path_to_model> --text "Your input text here"
```

## Implemented Architectures

This project includes the following transformer-based architectures:

*   **Transformer (Base)**: The original transformer model from "Attention Is All You Need".
*   **BERT**: Bidirectional Encoder Representations from Transformers.
*   **GPT**: Generative Pre-trained Transformer.
*   **ALBERT**: A Lite BERT for Self-supervised Learning of Language Representations.
*   **RoBERTa**: A Robustly Optimized BERT Pretraining Approach.
