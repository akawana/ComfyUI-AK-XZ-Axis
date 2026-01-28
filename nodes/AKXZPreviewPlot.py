import json
from typing import Any, Dict, List, Optional, Tuple

import torch

_AKXZ_MAGIC = b"AKXZ"
_HEADER_LEN = 8


def _u8_bytes_from_float01(x: torch.Tensor) -> bytes:
    x = torch.clamp(x * 255.0 + 0.5, 0, 255).to(torch.uint8).cpu()
    return x.numpy().tobytes()


def _extract_obj(img3: torch.Tensor) -> Optional[Dict[str, Any]]:
    if not isinstance(img3, torch.Tensor) or img3.ndim != 3 or img3.shape[-1] < 3:
        return None
    h, w = int(img3.shape[0]), int(img3.shape[1])
    if h <= 0 or w <= 0:
        return None
    row = img3[0, :, 0:3].reshape(-1)
    raw = _u8_bytes_from_float01(row)
    if len(raw) < _HEADER_LEN or raw[:4] != _AKXZ_MAGIC:
        return None
    size = int.from_bytes(raw[4:8], "big")
    if size <= 0 or (8 + size) > len(raw):
        return None
    payload = raw[8:8+size]
    try:
        obj = json.loads(payload.decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class AKXZPreviewPlot:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plot_image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("plot_image", "params_text")
    FUNCTION = "run"
    CATEGORY = "AK/Debug"

    def run(self, plot_image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        if not isinstance(plot_image, torch.Tensor) or plot_image.ndim != 4:
            raise ValueError("Expected plot_image as IMAGE tensor [B,H,W,C].")

        b = int(plot_image.shape[0])
        image_entries: List[Dict[str, Any]] = []

        for i in range(b):
            obj = _extract_obj(plot_image[i])
            if obj is None:
                continue

            # two possible cases:
            # 1) separate/batch: obj is exactly cfg.image[i]
            # 2) plot mode: obj is {"image":[...]}
            if "image" in obj and isinstance(obj["image"], list):
                image_entries.extend(obj["image"])
            else:
                image_entries.append(obj)

        result_obj = {"image": image_entries} if image_entries else {"image": []}

        text = json.dumps(result_obj, ensure_ascii=False, indent=2)

        return {
            "ui": {"text": [text]},
            "result": (plot_image, text),
        }


NODE_CLASS_MAPPINGS = {
    "AKXZPreviewPlot": AKXZPreviewPlot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AKXZPreviewPlot": "AKXZ Preview Plot",
}
