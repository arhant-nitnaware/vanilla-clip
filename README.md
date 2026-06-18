# Research-Grade OpenAI CLIP Framework

A comprehensive, offline-first implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training) designed for research experimentation. Features include experiment tracking, checkpoint management, mixed precision training, and comprehensive evaluation metrics.

## Features

- **Offline-First Architecture**: Models are cached locally with online fallback
- **Experiment Tracking**: Integrated TensorBoard and WandB support
- **Checkpoint Management**: Automatic versioning with best model tracking
- **Research-Grade Training**: Mixed precision, gradient accumulation, learning rate scheduling
- **Comprehensive Evaluation**: Recall@K, Precision@K, MRR, Mean Rank metrics
- **Flexible Inference**: Image-text similarity, zero-shot classification, retrieval
- **Configuration Management**: YAML-based experiment configurations
- **Reproducibility**: Seed control and deterministic training

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd openAI-CLIP

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package (optional)
pip install -e .
```

## CLI Interface Usage

The CLI provides a comprehensive interface for training, evaluation, and inference with CLIP models. All commands support `--help` for detailed information.

### Overview

```bash
python -m clip_cli.cli --help
```

This shows all available commands: `train`, `eval`, `infer`, `config`, and `model`.

---

### Model Management

#### List All Available Models

View all available CLIP models with their sizes:

```bash
python -m clip_cli.cli model --list
```

**Output:**
```
Available models:
  RN50                 102.4 MB
  RN101                174.1 MB
  RN50x4               345.3 MB
  RN50x16              1238.6 MB
  RN50x64              4705.2 MB
  ViT-B-32             338.0 MB
  ViT-B-16             568.0 MB
  ViT-L-14             755.5 MB
  ViT-L-14-336px       1355.0 MB
```

#### View Cached Models

Check which models are already downloaded and cached locally:

```bash
python -m clip_cli.cli model --cached
```

**Output:**
```
Cached models:
  RN50                 244.0 MB - cache/models/RN50.pt
```

#### Download a Model

Pre-download a specific model (optional - models download automatically on first use):

```bash
python -m clip_cli.cli model --download ViT-B-32
```

This downloads the model to `cache/models/` with checksum verification.

#### Clear Model Cache

Remove all cached models (use with caution):

```bash
python -m clip_cli.cli model --clear-cache
```

#### Model Command Help

```bash
python -m clip_cli.cli model --help
```

---

### Configuration Management

#### List Available Configurations

View all configuration files in the `configs/` directory:

```bash
python -m clip_cli.cli config --list
```

**Output:**
```
Available configs:
  default.yaml
  test_experiment.yaml
```

#### Create a New Configuration

Create a new experiment configuration from the default template:

```bash
python -m clip_cli.cli config --create my_experiment
```

This creates `configs/my_experiment.yaml` which you can customize.

#### Configuration Command Help

```bash
python -m clip_cli.cli config --help
```

---

### Training

#### Basic Training

Train a CLIP model using a configuration file:

```bash
python -m clip_cli.cli train --config configs/default.yaml --data path/to/dataset
```

**Arguments:**
- `--config CONFIG` (required): Path to YAML configuration file
- `--data DATA`: Override dataset path from config
- `--resume RESUME`: Path to checkpoint to resume training from
- `--offline`: Run in offline mode (no downloads, use cached models only)

#### Resume Training

Resume training from a checkpoint:

```bash
python -m clip_cli.cli train --config configs/default.yaml --resume experiments/checkpoints/my_exp/checkpoint_epoch_5.pt
```

#### Offline Training

Train without internet access (requires pre-downloaded models):

```bash
python -m clip_cli.cli train --config configs/default.yaml --offline
```

#### Training Command Help

```bash
python -m clip_cli.cli train --help
```

---

### Evaluation

#### Basic Evaluation

Evaluate a trained model on validation set:

```bash
python -m clip_cli.cli eval --config configs/default.yaml --checkpoint experiments/checkpoints/my_exp/best.pt
```

**Arguments:**
- `--config CONFIG` (required): Path to YAML configuration file
- `--data DATA`: Override dataset path from config
- `--checkpoint CHECKPOINT`: Path to checkpoint file to evaluate
- `--split {train,val,test}`: Dataset split to evaluate (default: val)

#### Evaluate on Test Set

```bash
python -m clip_cli.cli eval --config configs/default.yaml --checkpoint experiments/checkpoints/my_exp/best.pt --split test
```

#### Evaluation Command Help

```bash
python -m clip_cli.cli eval --help
```

---

### Inference

#### Image-Text Similarity

Compute cosine similarity between an image and text:

```bash
python -m clip_cli.cli infer --model ViT-B-32 --similarity --image path/to/image.jpg --text "a dog"
```

**Output:**
```
Similarity: 0.2534
```

#### Zero-Shot Classification

Classify an image using text labels:

```bash
python -m clip_cli.cli infer --model ViT-B-32 --classify --image path/to/image.jpg --labels labels.txt --top-k 5
```

**labels.txt format:**
```
a dog
a cat
a bird
a car
a horse
```

**Output:**
```
Classification Results:
  a dog                         0.2534
  a horse                       0.2069
  a cat                         0.1929
```

#### Text-to-Image Retrieval

Retrieve most similar images for a text query:

```bash
python -m clip_cli.cli infer --model ViT-B-32 --retrieve --text "a dog" --images-dir path/to/images --top-k 5
```

**Output:**
```
Retrieval Results:
  path/to/images/dog1.jpg       0.2534
  path/to/images/dog2.jpg       0.2345
  path/to/images/dog3.jpg       0.2123
```

#### Image-to-Text Retrieval

Retrieve most similar texts for an image:

```bash
python -m clip_cli.cli infer --model ViT-B-32 --retrieve --image path/to/image.jpg --text "a dog" --top-k 5
```

#### Inference with Checkpoint

Use a fine-tuned checkpoint instead of pretrained model:

```bash
python -m clip_cli.cli infer --model ViT-B-32 --checkpoint experiments/checkpoints/my_exp/best.pt --classify --image path/to/image.jpg --labels labels.txt
```

#### Offline Inference

Run inference without internet access:

```bash
python -m clip_cli.cli infer --model ViT-B-32 --offline --similarity --image path/to/image.jpg --text "a dog"
```

#### Inference Command Help

```bash
python -m clip_cli.cli infer --help
```

---

### Quick Start Examples

#### 1. Setup and Explore

```bash
# List available models
python -m clip_cli.cli model --list

# Download a model (optional)
python -m clip_cli.cli model --download ViT-B-32

# Check cached models
python -m clip_cli.cli model --cached
```

#### 2. Create Configuration

```bash
# Create a new experiment config
python -m clip_cli.cli config --create my_first_experiment

# Edit the config file
nano configs/my_first_experiment.yaml
```

#### 3. Train a Model

```bash
# Train with default settings
python -m clip_cli.cli train --config configs/my_first_experiment.yaml --data path/to/my_dataset
```

#### 4. Evaluate the Model

```bash
# Evaluate on validation set
python -m clip_cli.cli eval --config configs/my_first_experiment.yaml --checkpoint experiments/checkpoints/my_first_experiment/best.pt
```

#### 5. Run Inference

```bash
# Zero-shot classification
python -m clip_cli.cli infer --model ViT-B-32 --classify --image path/to/test_image.jpg --labels labels.txt --top-k 3

# Image-text similarity
python -m clip_cli.cli infer --model ViT-B-32 --similarity --image path/to/test_image.jpg --text "a beautiful sunset"
```

---

### CLI Command Reference

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `model --list` | List all available models | - |
| `model --cached` | Show cached models | - |
| `model --download NAME` | Download a model | Model name |
| `model --clear-cache` | Clear model cache | - |
| `config --list` | List configurations | - |
| `config --create NAME` | Create new config | Experiment name |
| `train --config FILE` | Train model | Config, data, resume, offline |
| `eval --config FILE` | Evaluate model | Config, checkpoint, split |
| `infer --similarity` | Image-text similarity | Model, image, text, checkpoint |
| `infer --classify` | Zero-shot classification | Model, image, labels, top-k |
| `infer --retrieve` | Retrieval | Model, text/images-dir, top-k |

---

### Getting Help

For detailed help on any command:

```bash
python -m clip_cli.cli --help              # Main help
python -m clip_cli.cli train --help         # Training help
python -m clip_cli.cli eval --help          # Evaluation help
python -m clip_cli.cli infer --help         # Inference help
python -m clip_cli.cli config --help        # Config help
python -m clip_cli.cli model --help         # Model help
```

## Dataset Format

### Directory Structure

```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── train.csv
├── val/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── val.csv
└── test/
    ├── images/
    └── test.csv
```

### CSV Format

Each CSV file should have `image` and `text` columns:

```csv
image,text
images/image1.jpg,A dog running in the park
images/image2.jpg,A cat sleeping on a sofa
images/image3.jpg,A bird flying in the sky
```

## Configuration

### Creating a New Configuration

```bash
python -m clip_cli.cli config --create my_experiment
```

This creates `configs/my_experiment.yaml` that you can customize.

### Configuration Structure

```yaml
experiment:
  name: clip_experiment
  description: "Experiment description"
  seed: 42
  device: auto  # auto, cuda, cpu
  mixed_precision: true
  distributed: false
  world_size: 1

model:
  name: ViT-B-32  # RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B-32, ViT-B-16, ViT-L-14, ViT-L-14-336px
  pretrained: true
  freeze_image_encoder: false
  freeze_text_encoder: false
  checkpoint_path: null

data:
  dataset_path: data
  batch_size: 32
  num_workers: 4
  pin_memory: true
  image_size: 224
  train_split: train
  val_split: val
  test_split: test

training:
  epochs: 10
  learning_rate: 1.0e-05
  weight_decay: 0.01
  warmup_epochs: 2
  gradient_clip: 1.0
  accumulation_steps: 1
  log_interval: 10
  eval_interval: 1
  save_interval: 1
  early_stopping_patience: 5
  optimizer: adamw  # adamw, adam
  scheduler: cosine  # cosine, linear

evaluation:
  batch_size: 64
  num_workers: 4
  metrics:
    - recall@1
    - recall@5
    - recall@10
    - precision@1
    - precision@5
    - mrr
    - mean_rank
```

## Experiment Tracking

### TensorBoard

```bash
tensorboard --logdir experiments/logs
```

### WandB Integration

To enable WandB tracking, modify your config or pass arguments:

```python
from clip_cli.core.logger import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="my_experiment",
    log_dir="experiments/logs",
    use_tensorboard=True,
    use_wandb=True,
    wandb_project="clip-research",
    wandb_entity="your-username",
)
```

## Checkpoint Management

### List Checkpoints

Checkpoints are automatically managed with versioning. The best checkpoint is always preserved.

### Resume Training

```bash
python -m clip_cli.cli train --config configs/default.yaml --resume experiments/checkpoints/clip_experiment/checkpoint_epoch_5_step_1000_20240101_120000.pt
```

### Load from Checkpoint

```python
from clip_cli.core.checkpoint import CheckpointManager

checkpoint_manager = CheckpointManager(experiment_name="my_experiment")
checkpoint_data = checkpoint_manager.load_best_checkpoint()

model.load_state_dict(checkpoint_data["model_state_dict"])
optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
```

## Advanced Features

### Mixed Precision Training

Automatically enabled with `mixed_precision: true` in config. Uses PyTorch AMP for faster training with reduced memory usage.

### Gradient Accumulation

Useful for larger effective batch sizes:

```yaml
training:
  batch_size: 16
  accumulation_steps: 4  # Effective batch size = 64
```

### Learning Rate Scheduling

Supports cosine annealing with warmup:

```yaml
training:
  learning_rate: 1.0e-05
  warmup_epochs: 2
  scheduler: cosine
```

### Freezing Encoders

Fine-tune only one encoder:

```yaml
model:
  freeze_image_encoder: true  # Freeze visual encoder
  freeze_text_encoder: false  # Train text encoder
```

## Offline Mode

For environments without internet access:

```bash
python -m clip_cli.cli train --config configs/default.yaml --offline
```

Pre-download models in online mode first:

```bash
python -m clip_cli.cli model --download ViT-B-32
python -m clip_cli.cli model --download ViT-L-14
```

## Project Structure

```
openAI-CLIP/
├── configs/              # Configuration files
│   └── default.yaml
├── experiments/          # Experiment outputs
│   ├── checkpoints/     # Model checkpoints
│   ├── logs/            # Training logs
│   ├── metrics/         # Evaluation metrics
│   └── embeddings/      # Saved embeddings
├── cache/               # Model cache
│   ├── models/          # Downloaded models
│   └── downloads/       # Temporary downloads
├── data/                # Datasets
├── src/
│   └── clip_cli/
│       ├── core/        # Core modules (config, logger, checkpoint, model_loader)
│       ├── modules/     # Training, evaluation, inference, dataset
│       ├── utils/       # Utility functions
│       └── cli.py       # Main CLI entry point
├── requirements.txt
├── setup.py
└── README.md
```

## Supported Models

- **RN50**: ResNet-50 (102 MB)
- **RN101**: ResNet-101 (174 MB)
- **RN50x4**: ResNet-50 4x (345 MB)
- **RN50x16**: ResNet-50 16x (1.2 GB)
- **RN50x64**: ResNet-50 64x (4.7 GB)
- **ViT-B-32**: Vision Transformer Base/32 (338 MB) - Default
- **ViT-B-16**: Vision Transformer Base/16 (568 MB)
- **ViT-L-14**: Vision Transformer Large/14 (755 MB)
- **ViT-L-14-336px**: Vision Transformer Large/14 336px (1.4 GB)

## Evaluation Metrics

- **Recall@K**: Fraction of correct matches in top-K results
- **Precision@K**: Average precision of top-K results
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of correct matches
- **Mean Rank**: Average rank of correct matches (lower is better)
- **Median Rank**: Median rank of correct matches (lower is better)

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Enable `mixed_precision: true`
- Use gradient accumulation with smaller batch size
- Use a smaller model (e.g., RN50 instead of ViT-L-14)

### Slow Training

- Increase `num_workers` for data loading
- Enable `mixed_precision: true`
- Use a smaller model
- Reduce image size

### Model Download Fails

- Check internet connection
- Use `--offline` mode if model is already cached
- Manually download model from OpenAI CDN and place in `cache/models/`

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{clip_cli_research,
  title={Research-Grade OpenAI CLIP Framework},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo/openAI-CLIP}
}
```

## License

MIT License

## Acknowledgments

- OpenAI for the original CLIP model and implementation
- PyTorch team for the deep learning framework
- Hugging Face for various utilities
