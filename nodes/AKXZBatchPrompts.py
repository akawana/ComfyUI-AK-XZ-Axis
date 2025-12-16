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


def _encode_cached(clip: Any, text: str, cache: Dict[str, Any]) -> Any:
    if text in cache:
        return cache[text]
    if clip is None:
        raise ValueError("clip is required")

    conditioning: Any

    if hasattr(clip, "tokenize") and hasattr(clip, "encode_from_tokens"):
        tokens = clip.tokenize(text)
        enc = clip.encode_from_tokens(tokens, return_pooled=True)
        if isinstance(enc, (tuple, list)) and len(enc) >= 2:
            cond, pooled = enc[0], enc[1]
        else:
            cond, pooled = enc, None
        meta: Dict[str, Any] = {}
        if pooled is not None:
            meta["pooled_output"] = pooled
        conditioning = [[cond, meta]]
    else:
        enc = clip.encode(text)
        if isinstance(enc, (tuple, list)) and len(enc) >= 2:
            cond, pooled = enc[0], enc[1]
            conditioning = [[cond, {"pooled_output": pooled}]]
        else:
            conditioning = enc

    cache[text] = conditioning
    return conditioning


class AKXZBatchPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP",),
            "pos_0": ("STRING", {"forceInput": True}),
            "neg_0": ("STRING", {"forceInput": True}),
        }
        optional = {
            "xz_config": ("STRING", {"forceInput": True}),
        }
        for i in range(1, 11):
            optional[f"pos_{i}"] = ("STRING", {"forceInput": True})
            optional[f"neg_{i}"] = ("STRING", {"forceInput": True})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positives", "negatives", "xz_config")
    FUNCTION = "run"
    CATEGORY = "AK/XZ Axis"

    OUTPUT_IS_LIST = (True, True, False)

    def run(self, clip, pos_0=None, neg_0=None, xz_config=None, **kwargs):
        pairs_pos: List[str] = []
        pairs_neg: List[str] = []

        def add_pair(p: Any, n: Any):
            if p is None or n is None:
                return
            pairs_pos.append(str(p))
            pairs_neg.append(str(n))

        add_pair(pos_0, neg_0)
        for i in range(1, 11):
            add_pair(kwargs.get(f"pos_{i}", None), kwargs.get(f"neg_{i}", None))

        real_steps = len(pairs_pos)

        cfg = _coerce_json(xz_config)
        z_axis = cfg is not None

        cache_pos: Dict[str, Any] = {}
        cache_neg: Dict[str, Any] = {}

        positives: List[Any] = []
        negatives: List[Any] = []

        prev_pos_text: Optional[str] = None
        prev_neg_text: Optional[str] = None
        prev_pos_cond: Any = None
        prev_neg_cond: Any = None

        for ptxt, ntxt in zip(pairs_pos, pairs_neg):
            if prev_pos_text is not None and ptxt == prev_pos_text:
                pcond = prev_pos_cond
            else:
                pcond = _encode_cached(clip, ptxt, cache_pos)

            if prev_neg_text is not None and ntxt == prev_neg_text:
                ncond = prev_neg_cond
            else:
                ncond = _encode_cached(clip, ntxt, cache_neg)

            positives.append(pcond)
            negatives.append(ncond)

            prev_pos_text = ptxt
            prev_neg_text = ntxt
            prev_pos_cond = pcond
            prev_neg_cond = ncond

        if z_axis:
            images = _get_images_list(cfg) if cfg is not None else []
            target_steps = len(images)

            if target_steps > 0 and len(positives) > 0:
                if len(positives) < target_steps:
                    last_p = positives[-1]
                    last_n = negatives[-1]
                    positives.extend([last_p] * (target_steps - len(positives)))
                    negatives.extend([last_n] * (target_steps - len(negatives)))
                elif len(positives) > target_steps:
                    positives = positives[:target_steps]
                    negatives = negatives[:target_steps]

            if cfg is None:
                cfg = {"image": []}
                images = cfg["image"]
            if "image" not in cfg or not isinstance(cfg.get("image"), list):
                cfg["image"] = images

            cap = max(0, real_steps - 1)
            for i in range(len(images)):
                if not isinstance(images[i], dict):
                    images[i] = {}
                j = i if i <= cap else cap
                ptxt = pairs_pos[j] if pairs_pos else ""
                ntxt = pairs_neg[j] if pairs_neg else ""
                images[i]["z_text"] = f"Pos: {ptxt}" if ntxt == "" else f"Pos: {ptxt} || Neg: {ntxt}"

            output_json = json.dumps(cfg, ensure_ascii=False)
            return (positives, negatives, output_json)

        images_out: List[Dict[str, Any]] = []
        for i in range(real_steps):
            images_out.append({"x_text": f"Pos: {pairs_pos[i]}" if pairs_neg[i] == "" else f"Pos: {pairs_pos[i]} || Neg: {pairs_neg[i]}"})
        cfg_out = {"image": images_out}
        output_json = json.dumps(cfg_out, ensure_ascii=False)
        return (positives, negatives, output_json)


NODE_CLASS_MAPPINGS = {
    "AK XZ Batch Prompts": AKXZBatchPrompts
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AK XZ Batch Prompts": "AK XZ Batch Prompts"
}
