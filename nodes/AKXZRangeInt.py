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


class AKXZRangeInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (["Seed", "Step"], {"default": "Seed"}),
                "start": ("INT", {"default": 0, "min": 0, "step": 1}),
                "end": ("INT", {"default": 1, "min": 1, "step": 1}),
                "steps": ("INT", {"default": 0, "min": 1, "step": 1}),
                "steps_define_end": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "xz_config": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("range", "xz_config")
    FUNCTION = "run"
    CATEGORY = "AK/XZ Axis"

    OUTPUT_IS_LIST = (True, False)

    def run(
        self,
        type: str,
        start: int,
        end: int,
        steps: int,
        steps_define_end: bool,
        xz_config: Any = None,
    ) -> Tuple[List[int], str]:
        cfg = _coerce_json(xz_config)

        zAxis = cfg is not None
        images_list: Optional[List[Dict[str, Any]]] = None
        if zAxis:
            images_list = _get_images_list(cfg) or []

        start_i = int(start)
        s_i = int(steps)
        if bool(steps_define_end):
            end_i = start_i + max(1, s_i) - 1
        else:
            end_i = int(end)

        count = abs(end_i - start_i) + 1
        if count < 1:
            count = 1

        if zAxis:
            desired = len(images_list) if images_list is not None else 0
            N = desired if desired > 0 else 1
        else:
            s = int(steps)
            N = max(1, s)

        outputRange: List[int] = []

        if start_i == end_i:
            outputRange = [end_i for _ in range(N)]
        else:
            N = min(N, count)
            if N == 1:
                outputRange = [end_i]
            else:
                direction = 1 if end_i >= start_i else -1
                span = abs(end_i - start_i)
                delta = math.floor(span / (N - 1))
                for i in range(N - 1):
                    outputRange.append(start_i + (direction * delta * i))
                outputRange.append(end_i)

        if zAxis:
            if cfg is None:
                cfg = {"image": []}
                images_list = cfg["image"]
            if images_list is None:
                images_list = []
                cfg["image"] = images_list

            for i in range(len(images_list)):
                v = outputRange[i] if i < len(outputRange) else outputRange[-1]
                images_list[i]["z_text"] = f"{type}: {v}"

            outputJson = json.dumps(cfg, ensure_ascii=False)
        else:
            images_out: List[Dict[str, Any]] = []
            for i in range(len(outputRange)):
                images_out.append({"x_text": f"{type}: {outputRange[i]}"})
            cfg_out = {"image": images_out}
            outputJson = json.dumps(cfg_out, ensure_ascii=False)

        return (outputRange, outputJson)


NODE_CLASS_MAPPINGS = {
    "AK XZ Range Int": AKXZRangeInt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AK XZ Range Int": "AK XZ Range Int"
}
