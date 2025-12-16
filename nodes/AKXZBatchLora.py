import json
from typing import Any, Dict, List, Optional


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


def _get_images_list(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = cfg.get("image")
    if isinstance(v, list):
        return [it if isinstance(it, dict) else {} for it in v]
    v = cfg.get("images")
    if isinstance(v, list):
        return [it if isinstance(it, dict) else {} for it in v]
    return []


class AKXZBatchLora:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model_0": ("MODEL",),
            "con_0": ("CONDITIONING",),
        }
        optional = {"xz_config": ("STRING", {"forceInput": True})}
        for i in range(1, 11):
            optional[f"model_{i}"] = ("MODEL",)
            optional[f"con_{i}"] = ("CONDITIONING",)
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("models", "conds", "xz_config")
    FUNCTION = "run"
    CATEGORY = "AK/XZ Axis"

    OUTPUT_IS_LIST = (True, True, False)

    def run(self, model_0=None, con_0=None, xz_config=None, **kwargs):
        models: List[Any] = []
        conds: List[Any] = []

        if model_0 is not None and con_0 is not None:
            models.append(model_0)
            conds.append(con_0)

        for i in range(1, 11):
            m = kwargs.get(f"model_{i}", None)
            c = kwargs.get(f"con_{i}", None)
            if m is not None and c is not None:
                models.append(m)
                conds.append(c)

        real_steps = len(models)
        pairs_count_pre = real_steps

        cfg = _coerce_json(xz_config)
        z_axis = cfg is not None

        if z_axis:
            images = _get_images_list(cfg) if cfg is not None else []
            real_steps = len(images)

            if real_steps > 0 and len(models) > 0:
                if len(models) < real_steps:
                    last_m = models[-1]
                    last_c = conds[-1]
                    models.extend([last_m] * (real_steps - len(models)))
                    conds.extend([last_c] * (real_steps - len(conds)))
                elif len(models) > real_steps:
                    models = models[:real_steps]
                    conds = conds[:real_steps]

            if cfg is None:
                cfg = {"image": []}
                images = cfg["image"]
            if "image" not in cfg or not isinstance(cfg.get("image"), list):
                cfg["image"] = images

            for i in range(len(images)):
                if not isinstance(images[i], dict):
                    images[i] = {}
                cap = pairs_count_pre - 1
                if cap < 0:
                    cap = 0
                j = i if i <= cap else cap
                images[i]["z_text"] = f"Lora: {j}"

            output_json = json.dumps(cfg, ensure_ascii=False)
            return (models, conds, output_json)

        images_out: List[Dict[str, Any]] = [{"x_text": f"Lora: {i}"} for i in range(real_steps)]
        cfg_out = {"image": images_out}
        output_json = json.dumps(cfg_out, ensure_ascii=False)
        return (models, conds, output_json)


NODE_CLASS_MAPPINGS = {
    "AK XZ Batch Lora": AKXZBatchLora
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AK XZ Batch Lora": "AK XZ Batch Lora"
}
