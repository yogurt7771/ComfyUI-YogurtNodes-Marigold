import json
import os
from pathlib import Path

import folder_paths
from huggingface_hub import hf_hub_download


V1_MODELS = {
    "marigold-depth-v1-1": "depth",
    "marigold-v1-0": "depth",
    "marigold-depth-lcm-v1-0": "depth",
    "marigold-normals-v1-1": "normals",
    "marigold-normals-v0-1": "normals",
    "marigold-normals-lcm-v0-1": "normals",
    "marigold-iid-appearance-v1-1": "appearance",
    "marigold-iid-lighting-v1-1": "lighting",
}
V2_MODELS = ["depth/Log-stage2", "depth/Log-stage1", "depth/Log-layered", "depth/Uniform-base", "depth/Uniform-layered", "depth/Disparity-base", "depth/Disparity-layered", "normals", "albedo"]
V2_REPO = "huawei-bayerlab/marigold-v2-0"
PROMPT_PREFIX = {"depth": "qwen_edit_2509_qwen_depth_realimg512", "normals": "qwen_edit_2509_qwen_normals_dummy512", "albedo": "qwen_edit_2509_qwen_albedo_rgb_dummy512"}
folder_paths.add_model_folder_path("marigold", os.path.join(folder_paths.models_dir, "Marigold"))


def read_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def contained_path(root, name):
    root = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(root / name))
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"Model path must stay within {root}: {name}")
    return candidate


def roots(version):
    found = [Path(p) / version for p in folder_paths.get_folder_paths("marigold")]
    if version == "v1":
        found = [Path(p) for p in folder_paths.get_folder_paths("diffusers")] + found
    return list(dict.fromkeys(found))


def list_v1_models():
    names = list(V1_MODELS)
    for root in roots("v1"):
        if root.is_dir():
            names.extend(p.name for p in root.iterdir() if p.is_dir() and (p / "model_index.json").is_file() and "marigold" in p.name.lower())
    return list(dict.fromkeys(names))


def component_weights(directory, component):
    stem = "model" if component == "text_encoder" else "diffusion_pytorch_model"
    for filename in (f"{stem}.fp16.safetensors", f"{stem}.safetensors", f"{stem}.fp16.bin", f"{stem}.bin"):
        path = contained_path(directory, f"{component}/{filename}")
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {component} weights in {directory}")


def ensure_v1(model_name, download_model):
    candidates = [contained_path(root, model_name) for root in roots("v1")]
    directory = next((p for p in candidates if (p / "model_index.json").is_file()), candidates[0])
    if download_model:
        if model_name not in V1_MODELS:
            raise ValueError("Downloads are only available for the listed official Marigold models.")
        files = ["model_index.json", "scheduler/scheduler_config.json"]
        for component in ("unet", "vae", "text_encoder"):
            files.append(f"{component}/config.json")
            try:
                component_weights(directory, component)
            except FileNotFoundError:
                stem = "model" if component == "text_encoder" else "diffusion_pytorch_model"
                files.append(f"{component}/{stem}.fp16.safetensors")
        for filename in files:
            if not contained_path(directory, filename).is_file():
                hf_hub_download(f"prs-eth/{model_name}", filename, local_dir=str(directory), token=False)
    if not (directory / "model_index.json").is_file():
        raise FileNotFoundError(f"Place {model_name} in models/Marigold/v1 or models/diffusers, or enable download_model.")
    for component in ("unet", "vae", "text_encoder"):
        component_weights(directory, component)
    return directory


def ensure_v2(checkpoint, download_model):
    if checkpoint not in V2_MODELS:
        raise ValueError(f"Unknown Marigold V2 checkpoint: {checkpoint}")
    modality = checkpoint.split("/")[0]
    filenames = [f"{checkpoint}/trainables.safetensors"] + [f"qwen_text_embeddings/{PROMPT_PREFIX[modality]}_{suffix}.pt" for suffix in ("prompt_embeds", "prompt_mask")]
    search_roots = roots("v2")
    paths = []
    for filename in filenames:
        path = next((contained_path(root, filename) for root in search_roots if contained_path(root, filename).is_file()), contained_path(search_roots[0], filename))
        if not path.is_file():
            if not download_model:
                raise FileNotFoundError(f"Missing {path}. Place the official V2 files in models/Marigold/v2 or enable download_model.")
            hf_hub_download(V2_REPO, filename, local_dir=str(search_roots[0]), token=False)
        paths.append(path)
    return paths
