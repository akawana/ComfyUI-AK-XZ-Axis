import json
import math
from typing import Any, Dict, List, Tuple, Optional


def _coerce_json(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _get_images_list(cfg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    for key in ("image", "images"):
        v = cfg.get(key)
        if isinstance(v, list):
            out: List[Dict[str, Any]] = []
            for item in v:
                if isinstance(item, dict):
                    out.append(item)
                else:
                    out.append({})
            return out
    for v in cfg.values():
        if isinstance(v, list) and v and all(isinstance(it, dict) for it in v):
            return v  # type: ignore[return-value]
    return None


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step


class AKXZRangeFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (["Cfg", "Denoise"], {"default": "Cfg"}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.05}),
                "end": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.05}),
                "steps": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
            "optional": {
                "xz_config": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("range", "xz_config")
    FUNCTION = "run"
    CATEGORY = "AK/XZ Axis"

    OUTPUT_IS_LIST = (True, False)

    def run(
        self,
        type: str,
        start: float,
        end: float,
        steps: int,
        xz_config: Any = None,
    ) -> Tuple[List[float], str]:
        cfg = _coerce_json(xz_config)

        zAxis = cfg is not None
        images_list: Optional[List[Dict[str, Any]]] = None
        if zAxis:
            images_list = _get_images_list(cfg) or []

        start_f = float(start)
        end_f = float(end)

        count = abs(end_f - start_f)

        if zAxis:
            desired = len(images_list) if images_list is not None else 0
            N = desired if desired > 0 else 1
        else:
            s = int(steps)
            N = max(1, s)

        outputRange: List[float] = []

        if start_f == end_f:
            outputRange = [end_f for _ in range(N)]
        else:
            if N <= 1:
                outputRange = [end_f]
            else:
                direction = 1.0 if end_f >= start_f else -1.0
                delta = abs(end_f - start_f) / float(N - 1)
                for i in range(N - 1):
                    outputRange.append(start_f + (direction * delta * float(i)))
                outputRange.append(end_f)

        outputRange = [_round_step(v, 0.05) for v in outputRange]

        if zAxis:
            if cfg is None:
                cfg = {"image": []}
                images_list = cfg["image"]
            if images_list is None:
                images_list = []
                cfg["image"] = images_list

            for i in range(len(images_list)):
                v = outputRange[i] if i < len(outputRange) else outputRange[-1]
                images_list[i]["z_text"] = f"{type}: {v:.2f}"

            outputJson = json.dumps(cfg, ensure_ascii=False)
        else:
            images_out: List[Dict[str, Any]] = []
            for i in range(len(outputRange)):
                images_out.append({"x_text": f"{type}: {outputRange[i]:.2f}"})
            cfg_out = {"image": images_out}
            outputJson = json.dumps(cfg_out, ensure_ascii=False)

        return (outputRange, outputJson)


NODE_CLASS_MAPPINGS = {
    "AK XZ Range Float": AKXZRangeFloat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AK XZ Range Float": "AK XZ Range Float"
}
