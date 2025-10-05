# src/engine.py

import math
from collections import defaultdict

import torch
from tqdm import tqdm

# Import COCO evaluation tools
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch, incorporating mixed-precision training and gradient clipping.

    Args:
        model: The neural network model to be trained.
        optimizer: The optimizer for updating model weights.
        data_loader: The DataLoader providing training data.
        device: The device (CPU or CUDA) to run the training on.
        epoch (int): The current epoch number.
    """
    model.train()  # Set the model to training mode

    # A GradScaler is used for mixed-precision training. It helps prevent numerical
    # underflow by scaling gradients, which can be very small when using float16.
    scaler = torch.amp.GradScaler("cuda")

    # Accumulator for various loss components to compute the average over the epoch.
    loss_accumulator = defaultdict(float)

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Training]")

    for i, (images, targets) in enumerate(progress_bar):
        # Move images and targets to the specified device.
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Use the autocast context manager for mixed-precision forward pass.
        # This automatically casts operations to lower-precision dtypes (like float16)
        # where appropriate to improve performance.
        with torch.amp.autocast("cuda"):
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

        # Sanity check: ensure the loss is a finite number. If it's NaN or infinity,
        # it can corrupt the model's weights, so we should skip this batch.
        if not math.isfinite(total_loss.item()):
            print(f"!!! Loss is {total_loss.item()}, stopping training at iteration {i} to prevent model corruption.")
            continue

        # --- Backward Pass ---
        optimizer.zero_grad()
        # scaler.scale() multiplies the loss by a scaling factor before backpropagation.
        scaler.scale(total_loss).backward()

        # Optional: Gradient clipping to prevent exploding gradients.
        # This caps the norm of the gradients to a maximum value (e.g., 1.0),
        # stabilizing training when gradients become too large.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # scaler.step() first unscales the gradients and then calls optimizer.step().
        scaler.step(optimizer)
        # Updates the scaling factor for the next iteration.
        scaler.update()

        # Record the loss values for logging.
        for k, v in loss_dict.items():
            loss_accumulator[k] += v.item()
        loss_accumulator["total_loss"] += total_loss.item()

        # Update the progress bar with the running average of the total loss.
        avg_loss = loss_accumulator["total_loss"] / (i + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    # Print the average losses for the entire epoch.
    num_batches = len(data_loader)
    print(f"Epoch {epoch + 1} training finished. Average losses:")
    for k, v in loss_accumulator.items():
        print(f"  - {k}: {v / num_batches:.4f}")

    avg_losses = {k: v / num_batches for k, v in loss_accumulator.items()}
    return avg_losses


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluates the model on a validation dataset using the COCO evaluation protocol.

    Args:
        model: The model to be evaluated.
        data_loader: The DataLoader for the validation data.
        device: The device to run evaluation on.

    Returns:
        coco_evaluator: An object containing the full evaluation results.
    """
    model.eval()  # Set the model to evaluation mode

    # Initialize COCO evaluation utilities.
    # `get_coco_api_from_dataset` creates a COCO-like API object from our dataset.
    # `CocoEvaluator` will manage the evaluation process.
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]  # We are evaluating bounding box detection.
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # ✨ NEW: Accumulator for validation losses.
    val_loss_accumulator = defaultdict(float)

    progress_bar = tqdm(data_loader, desc="Validation")

    for images, targets in progress_bar:
        images_gpu = [img.to(device) for img in images]

        # Using autocast during evaluation can also speed up inference.
        with torch.amp.autocast("cuda"):
            outputs = model(images_gpu)

        # ✨ NEW: Calculate validation loss.
        # Temporarily switch to train mode to get the loss dictionary.
        # This is safe within the @torch.no_grad() context, as no gradients are computed.
        model.train()
        targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast("cuda"):
            loss_dict = model(images_gpu, targets_gpu)
        model.eval()  # Switch back to evaluation mode immediately.

        for k, v in loss_dict.items():
            val_loss_accumulator[k] += v.item()
        val_loss_accumulator["total_loss"] += sum(loss for loss in loss_dict.values()).item()

        # Move model outputs to the CPU for post-processing and evaluation.
        outputs_cpu = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        # Format the results into the dictionary structure expected by the COCO evaluator:
        # {image_id: {boxes, labels, scores}}.
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs_cpu)}

        # Update the evaluator with the results from the current batch.
        coco_evaluator.update(res)

    # After iterating through all data, accumulate the results from all batches.
    coco_evaluator.accumulate()
    # Summarize the evaluation results and print them to the console.
    coco_evaluator.summarize()

    # ✨ NEW: Calculate and prepare the average validation losses for return.
    num_batches = len(data_loader)
    avg_val_losses = {f"val_{k}": v / num_batches for k, v in val_loss_accumulator.items()}

    return coco_evaluator, avg_val_losses
