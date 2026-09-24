from . import model_paths
from .v1 import depth_output, load_v1, normals_output, predict_v1
from .v2 import load_v2, predict_v2


DOWNLOAD = ("BOOLEAN", {"default": False, "tooltip": "Download only missing files for the selected official model. Disabled: local files only."})
SEED = ("INT", {"default": 2025, "min": 0, "max": 0xFFFFFFFFFFFFFFFF})


class YogurtMarigoldV1Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_name": (model_paths.list_v1_models(),),
            "dtype": (["auto", "float16", "bfloat16", "float32"], {"default": "auto"}),
            "download_model": DOWNLOAD,
        }}

    RETURN_TYPES = ("YOGURT_MARIGOLD_V1",)
    RETURN_NAMES = ("marigold",)
    FUNCTION = "load"
    CATEGORY = "YogurtMarigold/Loaders"
    DESCRIPTION = "Load Marigold V1 weights using native ComfyUI models. Searches models/Marigold/v1 and models/diffusers."

    def load(self, model_name, dtype="auto", download_model=False):
        return (load_v1(model_name, dtype, download_model),)


def v1_inputs():
    return {"required": {
        "marigold": ("YOGURT_MARIGOLD_V1",),
        "image": ("IMAGE",),
        "seed": SEED,
        "steps": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "0 uses the checkpoint's recommended denoising steps and scheduler."}),
        "ensemble_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        "processing_resolution": ("INT", {"default": 768, "min": 0, "max": 16384, "step": 8, "tooltip": "Longest inference edge. 0 uses input resolution. Output always matches the input size."}),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "Ensemble members processed together; increase if VRAM permits."}),
    }}


def run_v1(marigold, image, modality, **settings):
    if marigold.modality != modality:
        raise ValueError(f"This node requires a {modality} checkpoint; the loader selected {marigold.modality}.")
    return predict_v1(marigold, image, **settings)


class YogurtMarigoldV1Depth:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = v1_inputs()
        inputs["required"]["near_is_bright"] = ("BOOLEAN", {"default": True})
        return inputs

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth", "depth_mask")
    FUNCTION = "estimate"
    CATEGORY = "YogurtMarigold/V1"
    DESCRIPTION = "Relative depth (not metric distance). Supports V1.0, LCM and V1.1 checkpoints."

    def estimate(self, marigold, image, near_is_bright=True, **settings):
        return depth_output(run_v1(marigold, image, "depth", **settings), near_is_bright)


class YogurtMarigoldV1Normals:
    INPUT_TYPES = classmethod(lambda cls: v1_inputs())
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normals",)
    FUNCTION = "estimate"
    CATEGORY = "YogurtMarigold/V1"

    def estimate(self, marigold, image, **settings):
        return (normals_output(run_v1(marigold, image, "normals", **settings)),)


class YogurtMarigoldV1Appearance:
    INPUT_TYPES = classmethod(lambda cls: v1_inputs())
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("albedo", "roughness", "metallicity")
    FUNCTION = "decompose"
    CATEGORY = "YogurtMarigold/V1"
    DESCRIPTION = "Use marigold-iid-appearance-v1-1. Albedo is sRGB; roughness and metallicity are linear scalar maps."

    def decompose(self, marigold, image, **settings):
        result = run_v1(marigold, image, "appearance", **settings)
        return result[:, 0].movedim(1, -1), result[:, 1, 0].clone(), result[:, 1, 1].clone()


class YogurtMarigoldV1Lighting:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = v1_inputs()
        inputs["required"]["output_color_space"] = (["linear", "srgb"], {"default": "linear"})
        return inputs

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "shading", "residual")
    FUNCTION = "decompose"
    CATEGORY = "YogurtMarigold/V1"
    DESCRIPTION = "Use marigold-iid-lighting-v1-1. Linear output for compositing; sRGB applies Marigold's gamma 2.2 for preview."

    def decompose(self, marigold, image, output_color_space="linear", **settings):
        result = run_v1(marigold, image, "lighting", **settings)
        if output_color_space == "srgb":
            result = result.clamp_min(0).pow(1 / 2.2)
        return tuple(result[:, i].movedim(1, -1).contiguous() for i in range(3))


class YogurtMarigoldV2Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "Qwen Image Edit 2509 from ComfyUI's Load Diffusion Model or another compatible loader."}),
            "vae": ("VAE", {"tooltip": "Qwen Image VAE; use qwen_image_vae.safetensors."}),
            "checkpoint": (model_paths.V2_MODELS,),
            "download_model": DOWNLOAD,
        }}

    RETURN_TYPES = ("YOGURT_MARIGOLD_V2",)
    RETURN_NAMES = ("marigold",)
    FUNCTION = "load"
    CATEGORY = "YogurtMarigold/Loaders"
    DESCRIPTION = "Apply the official Marigold V2 adapter and fine-tuned VAE decoder to copies of a native Qwen MODEL and VAE. Downloads only the selected adapter and prompt embeddings."

    def load(self, model, vae, checkpoint, download_model=False):
        return (load_v2(model, vae, checkpoint, download_model),)


class YogurtMarigoldV2Estimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "marigold": ("YOGURT_MARIGOLD_V2",),
            "image": ("IMAGE",),
            "seed": SEED,
            "processing_resolution": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 16, "tooltip": "Maximum inference edge; 0 uses native size. Output matches input dimensions."}),
            "near_is_bright": ("BOOLEAN", {"default": True, "tooltip": "Depth display direction; accounts for disparity checkpoints. Does not change normals or albedo."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("prediction",)
    FUNCTION = "estimate"
    CATEGORY = "YogurtMarigold/V2"
    DESCRIPTION = "Single-step V2 depth, surface normals or sRGB albedo, selected by the loader checkpoint. Supports image batches."

    def estimate(self, marigold, image, seed=2025, processing_resolution=1024, near_is_bright=True):
        return (predict_v2(marigold, image, seed, processing_resolution, near_is_bright),)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (YogurtMarigoldV1Loader, YogurtMarigoldV1Depth, YogurtMarigoldV1Normals, YogurtMarigoldV1Appearance, YogurtMarigoldV1Lighting, YogurtMarigoldV2Loader, YogurtMarigoldV2Estimate)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YogurtMarigoldV1Loader": "Yogurt Marigold V1 Model Loader",
    "YogurtMarigoldV1Depth": "Yogurt Marigold V1 Depth",
    "YogurtMarigoldV1Normals": "Yogurt Marigold V1 Normals",
    "YogurtMarigoldV1Appearance": "Yogurt Marigold V1 Appearance",
    "YogurtMarigoldV1Lighting": "Yogurt Marigold V1 Lighting",
    "YogurtMarigoldV2Loader": "Yogurt Marigold V2 Model Loader",
    "YogurtMarigoldV2Estimate": "Yogurt Marigold V2 Estimate",
}
