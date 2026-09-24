# ComfyUI-YogurtNodes-Marigold

使用 ComfyUI 原生模型、优化注意力与显存管理运行 Marigold V1 / V2。无需安装其他 Marigold 节点、Diffusers 或 Transformers 运行时，不修改 `sys.path`；ComfyUI 自身原有依赖仍需正常安装。

## 节点

| 节点 | 用途与输出 |
| --- | --- |
| Yogurt Marigold V1 Model Loader | 加载 V1 模型，输出 `YOGURT_MARIGOLD_V1` |
| Yogurt Marigold V1 Depth | 相对深度 `IMAGE`、同值单通道 `MASK` |
| Yogurt Marigold V1 Normals | 法线 `IMAGE`，单位向量从 [-1,1] 映射到 [0,1] |
| Yogurt Marigold V1 Appearance | sRGB 反照率 `IMAGE`、线性粗糙度与金属度 `MASK` |
| Yogurt Marigold V1 Lighting | 反照率、光照、残差三个 `IMAGE`；可选择线性或 sRGB |
| Yogurt Marigold V2 Model Loader | 接入 Qwen MODEL / VAE，应用官方适配器与训练过的解码器 |
| Yogurt Marigold V2 Estimate | 按加载器所选模型输出深度、法线或 sRGB 反照率 `IMAGE` |

所有推理节点支持图片批次，输出恢复到输入尺寸。V2 深度需要 MASK 时连接 ComfyUI 的 `Convert Image to Mask`。

## 安装

将本目录放入 `ComfyUI/custom_nodes/`，在 **ComfyUI 使用的 Python 环境** 安装 `requirements.txt` 后重启。额外依赖仅为模型下载用的 `huggingface_hub`。OpenCV 已安装时，V2 输入缩放使用官方 Lanczos4；没有 OpenCV 时使用 PyTorch bicubic。

## 模型目录与下载

两个加载器均有 `download_model` 开关，默认 `false`，禁止隐式下载。设为 `true` 时仅补齐当前所选官方模型缺失的文件。模型直接保存到 ComfyUI 的模型目录，不使用 Hugging Face 全局权重缓存。

```text
ComfyUI/models/
├── diffusers/
│   └── marigold-depth-v1-1/          # 每种 V1 模型各自一个目录
│       ├── model_index.json
│       ├── scheduler/scheduler_config.json
│       ├── unet/config.json
│       ├── unet/diffusion_pytorch_model.fp16.safetensors
│       ├── vae/config.json
│       ├── vae/diffusion_pytorch_model.fp16.safetensors
│       ├── text_encoder/config.json
│       └── text_encoder/model.fp16.safetensors
├── diffusion_models/
│   └── qwen_image_edit_2509_fp8_e4m3fn_scaled.safetensors
├── vae/
│   └── qwen_image_vae.safetensors
└── Marigold/v2/
    ├── depth/Log-stage2/trainables.safetensors
    ├── normals/trainables.safetensors
    ├── albedo/trainables.safetensors
    └── qwen_text_embeddings/
        ├── qwen_edit_2509_qwen_depth_realimg512_prompt_embeds.pt
        └── qwen_edit_2509_qwen_depth_realimg512_prompt_mask.pt
```

V1 也识别标准无 `fp16` 后缀的 Safetensors，以及旧版 `.bin` 权重，优先使用 Safetensors。支持现有共享模型文件链接。V1 额外搜索 `models/Marigold/v1`。

`diffusers`、`diffusion_models`、`vae` 沿用 ComfyUI 的目录注册；`marigold` 注册根目录 `models/Marigold`。外置专用目录可在 `extra_model_paths.yaml` 配置：

```yaml
marigold_assets:
  base_path: E:/ComfyUI/models
  marigold: Marigold
  diffusers: diffusers
```

V2 的下载开关下载所选 Marigold 适配器及对应的两份预计算文本张量；Qwen 主模型和 VAE 由连接的标准加载器负责。无需下载 Qwen 文本编码器。V2 checkpoint 约 1.8 GB，Qwen 权重另计。

## V1 模型

官方仓库前缀为 `prs-eth/`：

- 深度：`marigold-v1-0`、`marigold-depth-lcm-v1-0`、`marigold-depth-v1-1`。
- 法线：`marigold-normals-v0-1`、`marigold-normals-lcm-v0-1`、`marigold-normals-v1-1`。
- 材质：`marigold-iid-appearance-v1-1`。
- 光照：`marigold-iid-lighting-v1-1`。

`steps=0` 使用 checkpoint 推荐步数；调度器按模型配置选择 DDIM 或 LCM。`ensemble_size` 增加多次估计，`batch_size` 控制同时运行的估计数量；深度结果先对齐尺度和偏移再合并。`processing_resolution=0` 使用原始尺寸，否则按长边缩放，补齐 VAE 所需倍数后裁掉补边。

材质分解的反照率已在 sRGB 空间，粗糙度和金属度不做 gamma。光照分解默认输出线性值，便于合成；sRGB 选项用官方 gamma 2.2 显示转换。光照与残差存在尺度不确定性。

## V2 模型

使用 **Qwen Image Edit 2509** 主模型与 **Qwen Image VAE**。标准 FP8 模型通过 `Load Diffusion Model` 加载；GGUF 可以通过已有的 GGUF 加载器提供 MODEL。请勿使用 Lightning 加速 LoRA 或 Qwen Image 2.1 主模型代替训练底模。

官方适配器仓库为 `huawei-bayerlab/marigold-v2-0`：

- `depth/Log-stage2`：默认模型，仿射不变的对数深度。
- `depth/Log-stage1`、`depth/Log-layered`。
- `depth/Uniform-base`、`depth/Uniform-layered`：线性深度。
- `depth/Disparity-base`、`depth/Disparity-layered`：视差。
- `normals`、`albedo`。

V2 固定单步推理，无需 KSampler。`seed` 控制 VAE 后验采样。`processing_resolution` 是输入长边上限，0 表示原始尺寸；推理尺寸向上对齐 16。`near_is_bright` 统一深度与视差的近亮远暗方向，不影响法线或反照率。输出深度经过逐图归一化，**不是以米为单位的距离**；Log 模型输出保留对数深度表达。

模型与 VAE 通过 ModelPatcher 克隆后应用补丁，不覆盖传入分支的原权重；计算和卸载由 ComfyUI 管理。显存紧张时降低分辨率、V1 的 `batch_size`，或使用 ComfyUI 支持的量化 Qwen 底模。

## 示例工作流

`example_workflows/` 包含 V1 深度、法线、材质、光照及 V2 原生加载示例。拖入 ComfyUI 后选择输入图与本地权重；需要下载时手动开启加载器开关。

## 参考

- [kijai/ComfyUI-Marigold](https://github.com/kijai/ComfyUI-Marigold)
- [visualbruno/ComfyUI-Marigold-v2](https://github.com/visualbruno/ComfyUI-Marigold-v2)
- [Manivannansvfx/ComfyUI-MarigoldV2](https://github.com/Manivannansvfx/ComfyUI-MarigoldV2)
- [Marigold V1](https://github.com/prs-eth/Marigold)
- [Marigold V2](https://github.com/huawei-bayerlab/marigold-v2)

本项目为独立集成，不附带或重新授权第三方模型权重；权重遵循其各自许可证。
