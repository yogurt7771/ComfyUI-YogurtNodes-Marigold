import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.clip_model
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.utils

from .imaging import ensemble_depth, normalize_depth, prepare_image, resize
from .model_paths import V1_MODELS, component_weights, ensure_v1, read_json
from .schedulers import MarigoldScheduler


@dataclass
class V1Model:
    model: object
    vae: object
    text_encoder: object
    config: dict
    scheduler_config: dict
    modality: str
    targets: int
    latent_scale: float


def load_v1(model_name, dtype="auto", download_model=False):
    directory = ensure_v1(model_name, download_model)
    config = read_json(directory / "model_index.json")
    unet_config = read_json(directory / "unet/config.json")
    names = config.get("target_properties", {}).get("target_names", [])
    modality = config.get("prediction_type", V1_MODELS.get(model_name, "depth"))
    if modality == "intrinsics":
        if names == ["albedo", "material"]:
            modality = "appearance"
        elif names == ["albedo", "shading", "residual"]:
            modality = "lighting"
        else:
            raise ValueError(f"Unsupported intrinsic targets: {names}")
    if modality not in ("depth", "normals", "appearance", "lighting"):
        raise ValueError(f"Unsupported Marigold V1 prediction: {modality}")
    # The released SD2 UNets differ in input/output channels, not block layout.
    layout = {"num_res_blocks": [2, 2, 2, 2], "channel_mult": [1, 2, 4, 4], "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0], "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], "transformer_depth_middle": 1}
    state = comfy.utils.load_torch_file(str(component_weights(directory, "unet")), safe_load=True)
    mapping = comfy.utils.unet_to_diffusers(layout)
    converted = {mapping[k]: v for k, v in state.items() if k in mapping}
    unmatched = set(state) - mapping.keys()
    if unmatched:
        raise ValueError(f"Unsupported Marigold UNet weights: {sorted(unmatched)[:3]}")
    options = {} if dtype == "auto" else {"dtype": {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]}
    model = comfy.sd.load_diffusion_model_state_dict(converted, model_options=options)
    if model is None:
        raise ValueError("Could not load the Marigold SD2 UNet with this ComfyUI version.")
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(component_weights(directory, "vae")), safe_load=True))
    vae.throw_exception_if_invalid()
    vae = copy.copy(vae)
    vae.process_output = lambda pixels: pixels

    text_config = read_json(directory / "text_encoder/config.json")
    text_dtype = mm.text_encoder_dtype(mm.text_encoder_device())
    text = comfy.clip_model.CLIPTextModel_(text_config, text_dtype, mm.text_encoder_offload_device(), comfy.ops.manual_cast)
    text_state = comfy.utils.load_torch_file(str(component_weights(directory, "text_encoder")), safe_load=True)
    text_state = {k.removeprefix("text_model."): v for k, v in text_state.items() if k != "text_model.embeddings.position_ids"}
    text.load_state_dict(text_state, strict=True)
    text_encoder = comfy.model_patcher.ModelPatcher(text, load_device=mm.text_encoder_device(), offload_device=mm.text_encoder_offload_device())
    return V1Model(model, vae, text_encoder, config, read_json(directory / "scheduler/scheduler_config.json"), modality, unet_config["out_channels"] // 4, read_json(directory / "vae/config.json").get("scaling_factor", 0.18215))


def empty_prompt(text_encoder):
    mm.load_models_gpu([text_encoder], force_full_load=True)
    device = text_encoder.load_device
    # CLIP's causal attention makes the first two tokens identical to unpadded
    # empty-text encoding. Native CLIP expects its full positional length.
    length = text_encoder.model.embeddings.position_embedding.weight.shape[0]
    tokens = torch.tensor([[49406] + [49407] * (length - 1)], device=device)
    return text_encoder.model(tokens)[0][:, :2].clone()


def predict_v1(handle, image, seed=0, steps=0, ensemble_size=1, processing_resolution=768, batch_size=1):
    model, vae = handle.model, handle.vae
    device = model.load_device
    dtype = model.model.get_dtype_inference()
    steps = steps or handle.config.get("default_denoising_steps", 4 if handle.scheduler_config["_class_name"] == "LCMScheduler" else 10)
    prompt = empty_prompt(handle.text_encoder).to(device=device, dtype=dtype)
    progress = comfy.utils.ProgressBar(image.shape[0] * ensemble_size * steps)
    completed = 0
    results = []
    for image_index in range(image.shape[0]):
        mm.throw_exception_if_processing_interrupted()
        pixels, (height, width) = prepare_image(image[image_index:image_index + 1], processing_resolution)
        image_latent = vae.encode(pixels.movedim(1, -1)).to(device=device, dtype=dtype) * handle.latent_scale
        predictions = []
        generator = torch.Generator(device=device).manual_seed((seed + image_index) % (2 ** 64))
        for start in range(0, ensemble_size, batch_size):
            count = min(batch_size, ensemble_size - start)
            latent = torch.randn((count, handle.targets * 4, *image_latent.shape[-2:]), device=device, dtype=dtype, generator=generator)
            schedule = MarigoldScheduler(handle.scheduler_config, steps, device)
            memory = model.memory_required([count, 4 * (1 + handle.targets), *image_latent.shape[-2:]])
            mm.load_models_gpu([model], memory_required=memory)
            model.pre_run()
            try:
                for index, timestep in enumerate(schedule.timesteps):
                    mm.throw_exception_if_processing_interrupted()
                    x = torch.cat((image_latent.expand(count, -1, -1, -1), latent), dim=1)
                    time = torch.full((count,), timestep, device=device, dtype=torch.long)
                    prediction = model.model.diffusion_model(x, time, context=prompt.expand(count, -1, -1), transformer_options=model.model_options.get("transformer_options", {}).copy())
                    latent = schedule.step(prediction, index, latent, generator).to(dtype)
                    completed += count
                    progress.update_absolute(completed)
            finally:
                model.cleanup()
            latent = latent.reshape(count * handle.targets, 4, *latent.shape[-2:])
            decoded = vae.decode(latent / handle.latent_scale).movedim(-1, 1).float()
            decoded = decoded[..., :height, :width]
            predictions.append(decoded.reshape(count, handle.targets, 3, height, width).cpu())
        decoded = torch.cat(predictions)
        if handle.modality == "depth":
            depth = (decoded[:, 0].mean(1, keepdim=True).clamp(-1, 1) + 1) * 0.5
            output = ensemble_depth(depth)
        elif handle.modality == "normals":
            normals = decoded[:, 0].clamp(-1, 1)
            if not handle.config.get("use_full_z_range", True):
                normals[:, 2] = (normals[:, 2] + 1) * 0.5
            output = F.normalize(normals, dim=1, eps=1e-6).mean(0, keepdim=True)
        else:
            output = ((decoded.clamp(-1, 1) + 1) * 0.5).mean(0)
        results.append(resize(output, image.shape[1], image.shape[2]))
    return torch.stack(results)


def depth_output(prediction, near_is_bright):
    depth = normalize_depth(prediction[:, 0], near_is_bright)
    return depth.repeat(1, 3, 1, 1).movedim(1, -1), depth[:, 0]


def normals_output(prediction):
    return (F.normalize(prediction[:, 0], dim=1, eps=1e-6).movedim(1, -1) + 1) * 0.5
