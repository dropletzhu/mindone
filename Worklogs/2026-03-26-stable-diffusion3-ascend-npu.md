# 2026-03-26: Stable Diffusion 3 在 Ascend NPU 上的运行验证

## 背景
验证 MindONE 仓库中的 Stable Diffusion 3 能否在 Ascend NPU 上正常运行。

## 环境检查

| 检查项 | 结果 |
|--------|------|
| MindSpore版本 | ✅ 2.8.0 已安装 |
| Ascend CANN版本 | ✅ 8.5.0 已安装 |
| NPU设备 | ✅ 910B4 正常 (HBM 32GB) |
| 网络 (HuggingFace) | ❌ 无法访问 |
| 网络 (ModelScope) | ✅ 正常 |

## 工作内容

### 1. 模型下载
- 使用 ModelScope 下载模型: `AI-ModelScope/stable-diffusion-3-medium-diffusers`
- 下载位置: `/workspace/xql/mindone/models/AI-ModelScope/stable-diffusion-3-medium-diffusers`

### 2. 模型配置修复
- **问题**: ModelScope 下载的模型缺少 `model_index.json`
- **解决**: 手动创建 `model_index.json`，包含 mindone 的类路径

### 3. Tokenizer 文件修复
- **问题**: tokenizer 目录缺少 `vocab.json`, `tokenizer_config.json`, `special_tokens_map.json`
- **解决**: 从 `AI-ModelScope/clip-vit-large-patch14` 复制 tokenizer 文件到:
  - `tokenizer/`
  - `tokenizer_2/`

### 4. 运行时配置
- 需要设置 `ms.set_context(device_target="Ascend", device_id=0)` 使用NPU
- 需要跳过 T5 编码器以节省内存 (`text_encoder_3=None, tokenizer_3=None`)

## 测试结果

| 测试项 | 结果 |
|--------|------|
| Pipeline 加载 | ✅ 约 220 秒 |
| 推理 (512x512, 20步) | ✅ 约 47 秒 |
| 推理 (1024x1024, 28步) | ❌ OOM |
| 完整加载 (含T5) | ❌ OOM |

## 最终可用代码

```python
import mindspore as ms
ms.set_context(device_target="Ascend", device_id=0)

from mindone.diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    '/workspace/xql/mindone/models/AI-ModelScope/stable-diffusion-3-medium-diffusers',
    text_encoder_3=None,  # 跳过T5以节省内存
    tokenizer_3=None,
    mindspore_dtype=ms.float16,
)

prompt = "A cat holding a sign that says 'Hello MindSpore'"
image = pipe(prompt, num_inference_steps=20, height=512, width=512)[0][0]
image.save("sd3.png")
```

## 待优化项
1. T5 编码器内存占用大 (~4.7B 参数)，需要更大内存设备
2. 推理速度可进一步优化

## 输出文件
- 模型: `/workspace/xql/mindone/models/AI-ModelScope/stable-diffusion-3-medium-diffusers/`
- 测试图像: `/workspace/xql/mindone/sd3.png` (512x512)
