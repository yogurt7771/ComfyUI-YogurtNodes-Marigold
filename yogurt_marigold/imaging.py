from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

try:
    import cv2
except ImportError:
    cv2 = None


def resize(image, height, width, mode="bilinear"):
    if image.shape[-2:] == (height, width):
        return image
    return F.interpolate(image, (height, width), mode=mode, align_corners=False, antialias=mode == "bicubic")


def resize_v2(image, height, width):
    if image.shape[-2:] == (height, width):
        return image
    if cv2 is None:
        return resize(image, height, width, "bicubic")
    pixels = image.detach().float().cpu().movedim(1, -1).numpy()
    resized = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4) for frame in pixels])
    return torch.from_numpy(resized).movedim(-1, 1).to(device=image.device, dtype=image.dtype)


def prepare_image(image, resolution, multiple=8):
    pixels = image[..., :3].movedim(-1, 1)
    if pixels.shape[1] == 1:
        pixels = pixels.expand(-1, 3, -1, -1)
    height, width = pixels.shape[-2:]
    if resolution > 0:
        scale = resolution / max(height, width)
        height, width = max(1, int(height * scale)), max(1, int(width * scale))
        pixels = resize(pixels, height, width, "bicubic")
    padding = (0, (-width) % multiple, 0, (-height) % multiple)
    return F.pad(pixels, padding, mode="replicate"), (height, width)


def normalize_depth(depth, near_is_bright, disparity=False):
    low = depth.amin(dim=(-2, -1), keepdim=True)
    span = (depth.amax(dim=(-2, -1), keepdim=True) - low).clamp_min(1e-6)
    depth = (depth - low) / span
    return 1 - depth if near_is_bright != disparity else depth


def ensemble_depth(predictions):
    if predictions.shape[0] == 1:
        return predictions
    # Fit each affine-invariant prediction to a common range before reducing.
    source = predictions.detach().float().cpu().numpy()
    count = len(source)
    low = source.min(axis=(1, 2, 3))
    scales = 1 / np.maximum(source.max(axis=(1, 2, 3)) - low, 1e-6)
    initial = np.concatenate((scales, -low * scales)).astype(np.float64)
    stride = max(1, max(source.shape[-2:]) // 256)
    small = source[..., ::stride, ::stride]

    def cost(params):
        aligned = small * params[:count, None, None, None] + params[count:, None, None, None]
        error = sum(np.sqrt(np.mean((aligned[i] - aligned[j]) ** 2)) for i, j in combinations(range(count), 2))
        center = np.median(aligned, axis=0)
        return error + 0.02 * (abs(center.min()) + abs(1 - center.max()))

    params = minimize(cost, initial, method="BFGS", tol=1e-3, options={"maxiter": 2}).x
    aligned = source * params[:count, None, None, None] + params[count:, None, None, None]
    return torch.from_numpy(np.median(aligned, axis=0, keepdims=True).astype(np.float32)).to(predictions.device)


def v2_output(decoded, modality, near_is_bright=True, disparity=False):
    if modality == "depth":
        depth = normalize_depth(decoded.mean(dim=1, keepdim=True), near_is_bright, disparity)
        return depth.repeat(1, 3, 1, 1).movedim(1, -1)
    if modality == "normals":
        return (F.normalize(decoded, dim=1, eps=1e-6).movedim(1, -1) + 1) * 0.5
    if modality == "albedo":
        return ((decoded.movedim(1, -1) + 1) * 0.5).clamp_min(0).pow(1 / 2.2).clamp_max(1)
    raise ValueError(f"Unknown Marigold modality: {modality}")
