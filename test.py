from cmd import PROMPT
import sys 
import time
from pathlib import Path

import torch
from PIL import image
from diffusers import AutoModel,DiffusionPipeline,TorchAoConfig
from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
def _safe_has_compatible_shallow_copy_type(t1,t2):
    return True
torch._has_compatible_shallow_copy_type = _safe_has_compatible_shallow_copy_type
AffineQuantizedTensor.__torch_function__ = torch._C._disabled_torch_function_impl


def maini():
    model_dir = Path("./models")
    input_paths = [Path("./masked_person.jpeg"), Path("./cloth.png")]
    out_path = Path("qwen_edit_test.png")
    lora_path = Path("./Qwen-Image-Lightning-4steps-V2.0.safetensors")
    prompt = "by using the green masked area from Picture 1 as a reference for position , place the garment from Picture 2 on the person from Picture 1"
    negative_prompt = "ugly"
    steps = 4
    seed = 724723345395306

      # sanity checks
    if not model_dir.exists():
        print(f"[ERR] model_dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)
    for pth in input_paths:
        if not pth.is_file():
            print(f"[ERR] input image not found: {pth}", file=sys.stderr)
            sys.exit(2)
    if lora_path and not lora_path.is_file():
        print(f"[ERR] lora_path not found: {lora_path}", file=sys.stderr)
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] model_dir={model_dir} lora={lora_path} inputs={input_paths}")
    t0 = time.time()

    # TorchAO quantization config
    torch_dtype = torch.bfloat16
    quantization_config = TorchAoConfig("int8wo")

    # Quantized transformer backbone
    transformer = AutoModel.from_pretrained(
        str(model_dir),
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )

    # Build pipeline
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        transformer=transformer,
        torch_dtype=torch_dtype,
    )

    if lora_path:
        pipe.load_lora_weights(str(lora_path))

    # Save VRAM
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(seed) if seed is not None else None

    # Load input images
    images = [Image.open(str(p)).convert("RGB") for p in input_paths]

    print("[INFO] editing…")
    kwargs = dict(
        image=images if len(images) > 1 else images[0],
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=steps,
        num_images_per_prompt=1,
    )

    with torch.inference_mode():
        result = pipe(**kwargs)
        img = result.images[0]

    img.save(out_path)
    dt = time.time() - t0
    print(f"[OK] saved → {out_path}  ({dt:.2f}s)")


if __name__ == "__main__":
    main()  