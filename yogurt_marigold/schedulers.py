import numpy as np
import torch


class MarigoldScheduler:
    """The DDIM (eta=0) and LCM schedules used by released Marigold V1 models."""

    def __init__(self, config, steps, device):
        self.config = config
        self.kind = config["_class_name"]
        if self.kind not in ("DDIMScheduler", "LCMScheduler"):
            raise ValueError(f"Unsupported Marigold scheduler: {self.kind}")
        count = config.get("num_train_timesteps", 1000)
        self.steps = steps
        if config.get("trained_betas") is not None:
            betas = torch.tensor(config["trained_betas"], dtype=torch.float32, device=device)
        elif config.get("beta_schedule", "scaled_linear") == "scaled_linear":
            betas = torch.linspace(config.get("beta_start", 0.00085) ** 0.5, config.get("beta_end", 0.012) ** 0.5, count, device=device) ** 2
        elif config["beta_schedule"] == "linear":
            betas = torch.linspace(config["beta_start"], config["beta_end"], count, device=device)
        else:
            raise ValueError(f"Unsupported beta schedule: {config['beta_schedule']}")
        if config.get("rescale_betas_zero_snr", False):
            roots = (1 - betas).cumprod(0).sqrt()
            first, last = roots[0].clone(), roots[-1].clone()
            roots = (roots - last) * first / (first - last)
            bars = roots.square()
            betas = 1 - torch.cat((bars[:1], bars[1:] / bars[:-1]))
        self.alphas = (1 - betas).cumprod(0)
        self.final_alpha = 1.0 if config.get("set_alpha_to_one", True) else self.alphas[0]
        self.ratio = count // steps
        if self.kind == "LCMScheduler":
            original = config.get("original_inference_steps", 50)
            if steps > original:
                raise ValueError(f"This LCM checkpoint supports at most {original} denoising steps.")
            origin = list(range(count // original - 1, count, count // original))[::-1]
            self.timesteps = [origin[int(i * len(origin) / steps)] for i in range(steps)]
        else:
            spacing = config.get("timestep_spacing", "leading")
            if steps > count:
                raise ValueError(f"DDIM supports at most {count} denoising steps.")
            if spacing == "trailing":
                self.timesteps = (np.round(np.arange(count, 0, -count / steps)) - 1).astype(int).tolist()
            elif spacing == "leading":
                self.timesteps = [i * self.ratio + config.get("steps_offset", 0) for i in reversed(range(steps))]
            elif spacing == "linspace":
                self.timesteps = np.linspace(0, count - 1, steps).round().astype(int).tolist()[::-1]
            else:
                raise ValueError(f"Unsupported timestep spacing: {spacing}")
        if config.get("thresholding", False):
            raise ValueError("Dynamic thresholding is not used by released Marigold checkpoints.")

    def step(self, prediction, index, sample, generator):
        timestep = self.timesteps[index]
        alpha = self.alphas[timestep]
        beta = 1 - alpha
        prediction_type = self.config.get("prediction_type", "epsilon")
        if prediction_type == "v_prediction":
            original = alpha.sqrt() * sample - beta.sqrt() * prediction
            epsilon = alpha.sqrt() * prediction + beta.sqrt() * sample
        elif prediction_type == "epsilon":
            original = (sample - beta.sqrt() * prediction) / alpha.sqrt()
            epsilon = prediction
        elif prediction_type == "sample":
            original = prediction
            epsilon = (sample - alpha.sqrt() * original) / beta.sqrt()
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")
        if self.config.get("clip_sample", False):
            limit = self.config.get("clip_sample_range", 1.0)
            original = original.clamp(-limit, limit)
        if self.kind == "LCMScheduler":
            scaled = timestep * self.config.get("timestep_scaling", 10.0)
            skip = 0.25 / (scaled ** 2 + 0.25)
            out = scaled / (scaled ** 2 + 0.25) ** 0.5
            denoised = out * original + skip * sample
            if index == len(self.timesteps) - 1:
                return denoised
            previous_alpha = self.alphas[self.timesteps[index + 1]]
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            return previous_alpha.sqrt() * denoised + (1 - previous_alpha).sqrt() * noise
        previous = timestep - self.ratio
        previous_alpha = self.alphas[previous] if previous >= 0 else self.final_alpha
        return previous_alpha ** 0.5 * original + (1 - previous_alpha) ** 0.5 * epsilon
