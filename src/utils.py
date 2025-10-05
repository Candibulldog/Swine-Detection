# src/utils.py


def collate_fn(batch):
    """
    A custom collate function for the DataLoader in object detection tasks.

    In object detection, each sample (image) can have a different number of
    associated ground truth bounding boxes. The default collate function in
    PyTorch tries to stack tensors, which would fail because the 'boxes' and
    'labels' tensors within each target dictionary have varying lengths.

    This function overrides the default behavior. Instead of trying to stack
    the samples, it simply groups them. It takes a list of (image, target)
    tuples and transposes it into a tuple of (all_images, all_targets).

    Args:
        batch (list): A list of tuples, where each tuple contains an
                      image tensor and its corresponding target dictionary.
                      Example: [(image1, target1), (image2, target2), ...]

    Returns:
        tuple: A tuple containing two elements:
               - A tuple of all image tensors: (image1, image2, ...)
               - A tuple of all target dictionaries: (target1, target2, ...)
    """
    # `zip(*batch)` effectively transposes the list of tuples.
    # For a batch = [(img1, tgt1), (img2, tgt2)],
    # `zip(*batch)` becomes `zip((img1, tgt1), (img2, tgt2))`,
    # which yields (img1, img2) and (tgt1, tgt2).
    # The final result is ((img1, img2, ...), (tgt1, tgt2, ...)).
    return tuple(zip(*batch))
