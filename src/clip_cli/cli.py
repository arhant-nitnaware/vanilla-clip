"""
Main CLI for Research-grade CLIP Framework
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core.config import ConfigManager, ExperimentConfig
from .core.logger import ExperimentLogger
from .core.checkpoint import CheckpointManager
from .core.model_loader import ModelLoader
from .modules.trainer import CLIPTrainer
from .modules.evaluator import CLIPEvaluator
from .modules.inference import CLIPInference
from .modules.dataset import create_train_val_dataloaders
from .utils.common import set_seed, get_device, print_device_info


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="clip-cli",
        description="""
Research-grade OpenAI CLIP Framework
==================================
Offline-first architecture with experiment tracking, checkpoint management,
and comprehensive evaluation metrics.

Available Models:
  RN50, RN101, RN50x4, RN50x16, RN50x64
  ViT-B-32 (default), ViT-B-16, ViT-L-14, ViT-L-14-336px

Use 'clip-cli model --list' to see all available models with details.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python -m clip_cli.cli model --list
  
  # Train a model
  python -m clip_cli.cli train --config configs/default.yaml --data data/my_dataset
  
  # Evaluate a model
  python -m clip_cli.cli eval --config configs/default.yaml --checkpoint experiments/checkpoints/my_exp/best.pt
  
  # Inference - similarity
  python -m clip_cli.cli infer --model ViT-B-32 --similarity --image path/to/image.jpg --text "a dog"
  
  # Inference - zero-shot classification
  python -m clip_cli.cli infer --model ViT-B-32 --classify --image path/to/image.jpg --labels labels.txt --top-k 5
  
  # Inference - text-to-image retrieval
  python -m clip_cli.cli infer --model ViT-B-32 --retrieve --text "a dog" --images-dir path/to/images
  
For more information on a specific command, use: clip-cli <command> --help
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute (train, eval, infer, config, model)",
        metavar="COMMAND"
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train CLIP model on a dataset",
        description="Train a CLIP model with experiment tracking, checkpointing, and logging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (required)"
    )
    train_parser.add_argument(
        "--data",
        type=str,
        help="Path to dataset directory (overrides config.data.dataset_path)"
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint file to resume training from"
    )
    train_parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (no model downloads, uses cached models only)"
    )
    
    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate CLIP model on a dataset",
        description="Evaluate a trained CLIP model using comprehensive metrics (Recall@K, Precision@K, MRR, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (required)"
    )
    eval_parser.add_argument(
        "--data",
        type=str,
        help="Path to dataset directory (overrides config.data.dataset_path)"
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file to load (overrides config.model.checkpoint_path)"
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val)"
    )
    
    # Infer command
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference with CLIP model",
        description="Run inference tasks: image-text similarity, zero-shot classification, or retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    infer_parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="Model name to use (default: ViT-B-32). Use 'clip-cli model --list' to see all models"
    )
    infer_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file to load (optional, uses pretrained if not specified)"
    )
    infer_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (default: auto, selects cuda if available)"
    )
    infer_parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (no model downloads, uses cached models only)"
    )
    
    # Inference modes
    infer_group = infer_parser.add_mutually_exclusive_group(required=True)
    infer_group.add_argument(
        "--similarity",
        action="store_true",
        help="Compute cosine similarity between an image and text"
    )
    infer_group.add_argument(
        "--classify",
        action="store_true",
        help="Perform zero-shot classification on an image using text labels"
    )
    infer_group.add_argument(
        "--retrieve",
        action="store_true",
        help="Perform retrieval: text-to-image or image-to-text"
    )
    
    # Inference arguments
    infer_parser.add_argument(
        "--image",
        type=str,
        help="Path to image file (required for similarity, classify, and image-to-text retrieval)"
    )
    infer_parser.add_argument(
        "--text",
        type=str,
        help="Text prompt (required for similarity and text-to-image retrieval)"
    )
    infer_parser.add_argument(
        "--labels",
        type=str,
        help="Path to text file containing class labels (one per line, required for classify mode)"
    )
    infer_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)"
    )
    infer_parser.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing images for text-to-image retrieval (required with --retrieve and --text)"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Manage experiment configurations",
        description="Create, list, and manage YAML configuration files for experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_parser.add_argument(
        "--create",
        type=str,
        help="Create a new config file from the default template (specify experiment name)"
    )
    config_parser.add_argument(
        "--list",
        action="store_true",
        help="List all available configuration files in the configs directory"
    )
    
    # Model command
    model_parser = subparsers.add_parser(
        "model",
        help="Manage CLIP models",
        description="List available models, download models, and manage model cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_parser.add_argument(
        "--list",
        action="store_true",
        help="List all available CLIP models with their sizes and descriptions"
    )
    model_parser.add_argument(
        "--cached",
        action="store_true",
        help="List all models currently cached in the local cache directory"
    )
    model_parser.add_argument(
        "--download",
        type=str,
        help="Download a specific model by name (e.g., ViT-B-32)"
    )
    model_parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the entire model cache (use with caution)"
    )
    
    return parser


def train_command(args):
    """Handle train command"""
    print("Training mode")
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Override data path if provided
    if args.data:
        config.data.dataset_path = args.data
    
    # Set seed
    set_seed(config.seed)
    print_device_info()
    
    # Initialize logger
    logger = ExperimentLogger(
        experiment_name=config.name,
        log_dir="experiments/logs",
        use_tensorboard=True,
        use_wandb=False,
    )
    logger.log_config(config.__dict__)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        experiment_name=config.name,
        checkpoint_dir="experiments/checkpoints",
    )
    
    # Load model
    model_loader = ModelLoader(offline_mode=args.offline)
    model, preprocess = model_loader.load_model(
        model_name=config.model.name,
        device=config.device,
    )
    
    # Create dataloaders
    train_loader, val_loader = create_train_val_dataloaders(
        data_path=config.data.dataset_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
    )
    
    # Initialize trainer
    trainer = CLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        model_config=config.model,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        device=config.device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_data = checkpoint_manager.load_checkpoint(args.resume)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        trainer.current_epoch = checkpoint_data["epoch"]
        trainer.global_step = checkpoint_data["step"]
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Train
    with logger:
        trainer.train()
    
    logger.close()


def eval_command(args):
    """Handle eval command"""
    print("Evaluation mode")
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Override data path if provided
    if args.data:
        config.data.dataset_path = args.data
    
    # Set seed
    set_seed(config.seed)
    print_device_info()
    
    # Initialize logger
    logger = ExperimentLogger(
        experiment_name=f"{config.name}_eval",
        log_dir="experiments/logs",
        use_tensorboard=True,
        use_wandb=False,
    )
    
    # Load model
    model_loader = ModelLoader()
    model, preprocess = model_loader.load_model(
        model_name=config.model.name,
        device=config.device,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint_manager = CheckpointManager(experiment_name=config.name)
        checkpoint_data = checkpoint_manager.load_checkpoint(args.checkpoint)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    
    # Create dataloader
    from .modules.dataset import CLIPDataset, create_dataloader
    
    dataset = CLIPDataset(
        data_path=config.data.dataset_path,
        split=args.split,
        image_size=config.data.image_size,
    )
    
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=config.evaluation.batch_size,
        num_workers=config.evaluation.num_workers,
        shuffle=False,
    )
    
    # Initialize evaluator
    evaluator = CLIPEvaluator(
        model=model,
        val_loader=dataloader,
        logger=logger,
        device=config.device,
    )
    
    # Evaluate
    with logger:
        results = evaluator.evaluate(metrics=config.evaluation.metrics)
    
    logger.close()
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


def infer_command(args):
    """Handle infer command"""
    print("Inference mode")
    
    # Load model
    model_loader = ModelLoader(offline_mode=args.offline)
    model, preprocess = model_loader.load_model(
        model_name=args.model,
        device=args.device,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        import torch
        checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint_data["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Initialize inference engine
    inference = CLIPInference(
        model=model,
        preprocess=preprocess,
        device=args.device,
    )
    
    if args.similarity:
        if not args.image or not args.text:
            print("Error: --image and --text required for similarity mode")
            sys.exit(1)
        
        similarity = inference.compute_similarity(args.image, args.text)
        print(f"Similarity: {similarity:.4f}")
    
    elif args.classify:
        if not args.image or not args.labels:
            print("Error: --image and --labels required for classification mode")
            sys.exit(1)
        
        # Load labels
        with open(args.labels, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        
        results = inference.zero_shot_classify(args.image, labels, top_k=args.top_k)
        
        print("\nClassification Results:")
        for result in results:
            print(f"  {result['label']:<30} {result['score']:.4f}")
    
    elif args.retrieve:
        if args.text and args.images_dir:
            # Text to image retrieval
            images_dir = Path(args.images_dir)
            image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            results = inference.retrieve_images(args.text, image_paths, top_k=args.top_k)
            
            print("\nRetrieval Results:")
            for result in results:
                print(f"  {result['image_path']:<50} {result['score']:.4f}")
        
        elif args.image and args.text:
            # Image to text retrieval (single text for now)
            results = inference.retrieve_texts(args.image, [args.text], top_k=args.top_k)
            
            print("\nRetrieval Results:")
            for result in results:
                print(f"  {result['text']:<30} {result['score']:.4f}")
        
        else:
            print("Error: --text and --images-dir (for text->image) or --image and --text (for image->text) required")
            sys.exit(1)


def config_command(args):
    """Handle config command"""
    config_manager = ConfigManager()
    
    if args.create:
        config = config_manager.create_experiment_config(args.create)
        print(f"Created config: configs/{args.create}.yaml")
    
    elif args.list:
        configs = list(Path("configs").glob("*.yaml"))
        print("Available configs:")
        for config in configs:
            print(f"  {config.name}")


def model_command(args):
    """Handle model command"""
    model_loader = ModelLoader()
    
    if args.list:
        models = model_loader.list_available_models()
        print("Available models:")
        for model in models:
            info = model_loader.get_model_info(model)
            print(f"  {model:<20} {info.size_mb:.1f} MB")
    
    elif args.cached:
        cached = model_loader.list_cached_models()
        print("Cached models:")
        for model in cached:
            print(f"  {model['name']:<20} {model['size_mb']:.1f} MB - {model['path']}")
    
    elif args.download:
        model_loader._download_model(args.download)
        print(f"Downloaded model: {args.download}")
    
    elif args.clear_cache:
        model_loader.clear_cache()
        print("Cleared model cache")


def main():
    """Main entry point"""
    parser = build_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "train":
            train_command(args)
        elif args.command == "eval":
            eval_command(args)
        elif args.command == "infer":
            infer_command(args)
        elif args.command == "config":
            config_command(args)
        elif args.command == "model":
            model_command(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
