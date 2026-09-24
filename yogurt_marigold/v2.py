import copy
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.latent_formats
import comfy.lora
import comfy.model_management as mm
import comfy.sd
import comfy.utils
from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
from comfy.ldm.wan.vae import WanVAE

from .imaging import resize, resize_v2, v2_output
from .model_paths import ensure_v2


@dataclass
class V2Model:
    model: object
    vae: object
    prompt_paths: tuple
    modality: str
    disparity: bool


def convert_decoder(state):
    """Convert the released Diffusers Qwen decoder names to native Wan names."""
    converted = {}
    residual = {"norm1": "residual.0", "conv1": "residual.2", "norm2": "residual.3", "conv2": "residual.6", "conv_shortcut": "shortcut"}
    direct = {"post_quant_conv": "conv2", "decoder.conv_in": "decoder.conv1", "decoder.conv_out": "decoder.head.2", "decoder.norm_out": "decoder.head.0"}
    for key, tensor in state.items():
        prefix, _, suffix = key.rpartition(".")
        if prefix in direct:
            target = f"{direct[prefix]}.{suffix}"
        elif key.startswith("decoder.mid_block.attentions.0."):
            target = key.replace("decoder.mid_block.attentions.0.", "decoder.middle.1.", 1)
        elif match := re.fullmatch(r"decoder.mid_block.resnets.(\d+).([^.]+).(.+)", key):
            index, layer, suffix = match.groups()
            target = f"decoder.middle.{2 * int(index)}.{residual[layer]}.{suffix}"
        elif match := re.fullmatch(r"decoder.up_blocks.(\d+).resnets.(\d+).([^.]+).(.+)", key):
            block, index, layer, suffix = match.groups()
            target = f"decoder.upsamples.{4 * int(block) + int(index)}.{residual[layer]}.{suffix}"
        elif match := re.fullmatch(r"decoder.up_blocks.(\d+).upsamplers.0.(.+)", key):
            block, suffix = match.groups()
            target = f"decoder.upsamples.{4 * int(block) + 3}.{suffix}"
        else:
            raise ValueError(f"Unrecognized Marigold V2 decoder weight: {key}")
        converted[target] = tensor
    return converted


def patch_models(model, vae, state):
    lora = {key.removeprefix("Diffuser."): value for key, value in state.items() if key.startswith("Diffuser.")}
    if not lora:
        raise ValueError("The V2 checkpoint contains no Diffuser LoRA weights.")
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(lora, key_map)
    expected = sum(".lora_A." in key for key in lora)
    if not loaded or len(loaded) != expected:
        raise ValueError(f"Marigold adapter mismatch: loaded {len(loaded)} of {expected} LoRA pairs. Use Qwen Image Edit 2509.")
    patched_model = model.clone()
    applied = patched_model.add_patches(loaded, strength_patch=1.0)
    if len(applied) != len(loaded):
        raise ValueError("The Qwen model did not accept all Marigold adapter patches.")

    patched_vae = copy.copy(vae)
    patched_vae.patcher = vae.patcher.clone()
    decoder = convert_decoder({key.removeprefix("VAE."): value for key, value in state.items() if key.startswith("VAE.")})
    target = vae.first_stage_model.state_dict()
    for key, value in decoder.items():
        if key not in target or target[key].shape != value.shape:
            raise ValueError(f"Marigold decoder weight does not match the Qwen VAE: {key}")
    if decoder:
        patched_vae.patcher.add_patches({key: ("set", (value,)) for key, value in decoder.items()}, strength_patch=1.0, strength_model=0.0)
    # Preserve the signed decoder output, including depth beyond [-1, 1].
    patched_vae.process_output = lambda pixels: pixels
    return patched_model, patched_vae


def load_v2(model, vae, checkpoint, download_model=False):
    if not isinstance(model.model.diffusion_model, QwenImageTransformer2DModel):
        raise ValueError("Marigold V2 requires a Qwen Image Edit 2509 MODEL.")
    if not isinstance(vae.first_stage_model, WanVAE) or vae.latent_channels != 16:
        raise ValueError("Marigold V2 requires the Qwen Image VAE (16 latent channels).")
    checkpoint_path, embeds_path, mask_path = ensure_v2(checkpoint, download_model)
    state = comfy.utils.load_torch_file(str(checkpoint_path), safe_load=True)
    model, vae = patch_models(model, vae, state)
    return V2Model(model, vae, (embeds_path, mask_path), checkpoint.split("/")[0], "Disparity-" in checkpoint)


def encode_image(vae, pixels, generator):
    memory = vae.memory_used_encode((pixels.shape[0], 3, 1, *pixels.shape[-2:]), vae.vae_dtype)
    mm.load_models_gpu([vae.patcher], memory_required=memory, force_full_load=vae.disable_offload)
    x = (pixels * 2 - 1).to(device=vae.device, dtype=vae.vae_dtype).unsqueeze(2)
    features = vae.first_stage_model.encoder(x, feat_cache=None, feat_idx=[0])
    mean, logvar = vae.first_stage_model.conv1(features).chunk(2, dim=1)
    std = (0.5 * logvar.clamp(-30, 20)).exp()
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    return mean + std * noise


def transformer_step(model, latents, prompt):
    device = model.load_device
    dtype = model.model.get_dtype_inference()
    mm.load_models_gpu([model], memory_required=model.memory_required(list(latents.shape)))
    x = latents.to(device=device, dtype=dtype)
    context = prompt[0].to(device=device, dtype=dtype)
    mask = prompt[1].to(device=device, dtype=torch.int32)
    # Training rounds 499 to bf16 before dividing, yielding 0.5, not 0.499.
    timestep = (torch.full((x.shape[0],), 499.0, device=device, dtype=torch.bfloat16) / 1000).to(dtype)
    model.pre_run()
    try:
        prediction = model.model.diffusion_model(x=x, timestep=timestep, context=context, attention_mask=mask, transformer_options=model.model_options.get("transformer_options", {}).copy())
    finally:
        model.cleanup()
    return latents - prediction.to(device=latents.device, dtype=latents.dtype)


def predict_v2(handle, image, seed=2025, processing_resolution=1024, near_is_bright=True):
    prompt = tuple(torch.load(path, map_location="cpu", weights_only=True)[:1].clone() for path in handle.prompt_paths)
    latent_format = comfy.latent_formats.Wan21()
    progress = comfy.utils.ProgressBar(image.shape[0] * 3)
    output = []
    for index in range(image.shape[0]):
        mm.throw_exception_if_processing_interrupted()
        pixels = image[index:index + 1, ..., :3].movedim(-1, 1)
        if pixels.shape[1] == 1:
            pixels = pixels.expand(-1, 3, -1, -1)
        height, width = pixels.shape[-2:]
        scale = min(1.0, processing_resolution / max(height, width)) if processing_resolution > 0 else 1.0
        target_h = max(16, (round(height * scale) + 15) // 16 * 16)
        target_w = max(16, (round(width * scale) + 15) // 16 * 16)
        pixels = resize_v2(pixels, target_h, target_w)
        generator = torch.Generator(device=handle.vae.device).manual_seed((seed + index) % (2 ** 64))
        latents = latent_format.process_in(encode_image(handle.vae, pixels, generator))
        progress.update_absolute(index * 3 + 1)
        mm.throw_exception_if_processing_interrupted()
        latents = transformer_step(handle.model, latents, prompt)
        progress.update_absolute(index * 3 + 2)
        mm.throw_exception_if_processing_interrupted()
        decoded = handle.vae.decode(latent_format.process_out(latents))[:, 0].movedim(-1, 1).float()
        result = v2_output(decoded, handle.modality, near_is_bright, handle.disparity)
        result = resize(result.movedim(-1, 1), height, width)
        if handle.modality == "normals":
            result = (F.normalize(result * 2 - 1, dim=1, eps=1e-6) + 1) * 0.5
        output.append(result.movedim(1, -1).cpu())
        progress.update_absolute(index * 3 + 3)
    return torch.cat(output)
