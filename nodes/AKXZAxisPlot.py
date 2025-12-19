import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

def _split_text_lines(text):
    if not text:
        return []
    if '||' in text:
        return [t.strip() for t in text.split('||') if t.strip()]
    return [text]



def _parse_xz_config(xz_config: Union[str, Dict[str, Any], List[Any], None]) -> List[Dict[str, Any]]:
    if xz_config is None:
        return []

    cfg: Any = xz_config
    if isinstance(cfg, str):
        s = cfg.strip()
        if not s:
            return []
        try:
            cfg = json.loads(s)
        except Exception:
            return []

    if isinstance(cfg, dict):
        if isinstance(cfg.get("images"), list):
            return [x for x in cfg["images"] if isinstance(x, dict)]
        if isinstance(cfg.get("image"), list):
            return [x for x in cfg["image"] if isinstance(x, dict)]
        numeric_keys: List[int] = []
        for k in cfg.keys():
            try:
                numeric_keys.append(int(k))
            except Exception:
                pass
        if numeric_keys:
            out: List[Dict[str, Any]] = []
            for k in sorted(numeric_keys):
                v = cfg.get(str(k), cfg.get(k))
                if isinstance(v, dict):
                    out.append(v)
            return out
        return [cfg]

    if isinstance(cfg, list):
        return [x for x in cfg if isinstance(x, dict)]

    return []


def _image_tensor_to_pil(img: torch.Tensor) -> Image.Image:
    if not isinstance(img, torch.Tensor):
        raise TypeError("Expected torch.Tensor IMAGE")

    x = img.detach().cpu().numpy()
    if x.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims [H,W,C], got {x.shape}")

    x = np.clip(x, 0.0, 1.0)

    if x.shape[2] == 4:
        x = x[:, :, :3]

    x8 = (x * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(x8, mode="RGB")


def _pil_to_image_tensor(pil_img: Image.Image) -> torch.Tensor:
    pil_img = pil_img.convert("RGB")
    x = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(x)


def setImageMeta(pil_img: Image.Image, meta: dict) -> Image.Image:
    if pil_img is None:
        return pil_img
    if not isinstance(pil_img, Image.Image):
        return pil_img
    if not isinstance(meta, dict):
        meta = {}

    try:
        meta_json = json.dumps(meta, ensure_ascii=False)
    except Exception:
        meta_json = "{}"

    pil_img.info["ak_meta"] = meta_json
    for k in ("seed", "steps", "cfg", "denoise", "lora", "pos", "neg"):
        v = meta.get(k, "")
        if v is None:
            v = ""
        pil_img.info[k] = str(v)

    return pil_img


def _meta_pick(cfg: Dict[str, Any], key: str) -> str:
    k = (key or "").strip().lower()
    if not k:
        return ""
    for axis in ("x", "z"):
        for idx in (0, 1):
            n = cfg.get(f"{axis}_parameter_name_{idx}", "")
            if n is None:
                n = ""
            if str(n).strip().lower() == k:
                v = cfg.get(f"{axis}_parameter_value_{idx}", "")
                if v is None:
                    return ""
                return str(v)
    return ""


def _build_meta(cfg: Any) -> Dict[str, str]:
    if not isinstance(cfg, dict):
        return {"seed": "", "steps": "", "cfg": "", "denoise": "", "lora": "", "pos": "", "neg": ""}

    return {
        "seed": _meta_pick(cfg, "seed"),
        "steps": _meta_pick(cfg, "steps"),
        "cfg": _meta_pick(cfg, "cfg"),
        "denoise": _meta_pick(cfg, "denoise"),
        "lora": _meta_pick(cfg, "lora"),
        "pos": _meta_pick(cfg, "pos"),
        "neg": _meta_pick(cfg, "neg"),
    }


def _load_font(sz: int) -> ImageFont.ImageFont:
    sz = int(max(1, sz))

    candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
        "segoeui.ttf",
        "tahoma.ttf",
    ]

    win_dir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if win_dir:
        fonts_dir = os.path.join(win_dir, "Fonts")
        candidates.extend([
            os.path.join(fonts_dir, "arial.ttf"),
            os.path.join(fonts_dir, "Arial.ttf"),
            os.path.join(fonts_dir, "segoeui.ttf"),
            os.path.join(fonts_dir, "tahoma.ttf"),
            os.path.join(fonts_dir, "calibri.ttf"),
        ])

    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ])

    for c in candidates:
        try:
            return ImageFont.truetype(c, sz)
        except Exception:
            continue

    return ImageFont.load_default()


def _add_header_and_text(
    pil_img: Image.Image, header_height: int, x_text: str, z_text: str, sting_trim: int, header_layout: dict | None = None
) -> Image.Image:
    w, h = pil_img.size
    header_height = int(max(0, header_height))
    out = Image.new("RGB", (w, h + header_height), (255, 255, 255))
    out.paste(pil_img, (0, header_height))

    x_text = (x_text or "").strip()
    z_text = (z_text or "").strip()

    if header_height <= 0 or (not x_text and not z_text):
        return out

    draw = ImageDraw.Draw(out)

    try:
        sting_trim = int(sting_trim)
    except Exception:
        sting_trim = 30
    sting_trim = max(30, sting_trim)

    def _trim_line(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        if len(s) > sting_trim:
            return s[:sting_trim] + "..."
        return s

    lines: List[Tuple[str, Tuple[int, int, int]]] = []
    if x_text:
        for _t in _split_text_lines(x_text):
            lines.append((_trim_line(_t), (0, 0, 0)))
    if z_text:
        for _t in _split_text_lines(z_text):
            lines.append((_trim_line(_t), (0, 0, 255)))

    max_h = max(1, header_height)
    max_w = max(1, w)

    use_cached = False
    font = None
    spacing = None
    y0 = None

    if header_layout is not None:
        _font = header_layout.get("font")
        _spacing = header_layout.get("spacing")
        _y0 = header_layout.get("y0")
        if _font is not None and _spacing is not None and _y0 is not None:
            font = _font
            spacing = float(_spacing)
            y0 = int(_y0)
            use_cached = True

    if not use_cached:
        spacing = max(max_h * 0.1, 6)

        base_sz = int(max(14, max_h * (0.65 if len(lines) <= 1 else 0.44)))
        sz = base_sz

        while sz > 6:
            test_font = _load_font(sz)

            widths = []
            heights = []

            for t, _fill in lines:
                bbox = draw.textbbox((0, 0), t, font=test_font)
                widths.append(bbox[2] - bbox[0])
                heights.append(bbox[3] - bbox[1])

            total_h = sum(heights) + (spacing * (len(lines) - 1))
            max_line_w = max(widths) if widths else 0

            if total_h <= max_h - 2 and max_line_w <= max_w - 6:
                font = test_font
                break

            sz -= 1

        if font is None:
            font = _load_font(max(6, base_sz))

        heights = []
        for t, _fill in lines:
            bbox = draw.textbbox((0, 0), t, font=font)
            heights.append(bbox[3] - bbox[1])

        total_text_h = sum(heights) + (spacing * (len(lines) - 1))
        y0 = max(0, (header_height - total_text_h) // 4)

        if header_layout is not None:
            header_layout["font"] = font
            header_layout["spacing"] = spacing
            header_layout["y0"] = y0

    line_sizes: List[Tuple[int, int]] = []
    for t, _fill in lines:
        bbox = draw.textbbox((0, 0), t, font=font)
        line_sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))

    y = y0
    for (t, fill), (tw, th) in zip(lines, line_sizes):
        x = max(0, (w - tw) // 2)
        draw.text((x, y), t, fill=fill, font=font, stroke_width=1, stroke_fill=(255, 255, 255))
        y += th + spacing

    return out


class AKXZAxisPlot:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "xz_config": ("STRING", {"forceInput": True}),
                "header_height_percent": ("INT", {"default": 25, "min": 5, "max": 50, "step": 1}),
                "grid_spacing": ("INT", {"default": 0, "min": 0, "step": 1}),
                "sting_trim": ("INT", {"default": 30, "min": 30, "step": 1}),
                "draw_headers": ("BOOLEAN", {"default": True}),
                "plot_type": (["plot", "separate"], {"default": "plot"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("plot_image", "xz_config")
    FUNCTION = "run"
    CATEGORY = "AK/XZ Axis"

    def run(
        self,
        image,
        xz_config,
        header_height_percent,
        grid_spacing,
        sting_trim,
        draw_headers,
                plot_type,
    ):
        def _first(v, default=None):
            if isinstance(v, list) and v:
                return v[0]
            return v if v is not None else default

        header_height_percent = int(_first(header_height_percent, 5))
        grid_spacing = int(_first(grid_spacing, 0))
        sting_trim = int(_first(sting_trim, 30))
        plot_type = str(_first(plot_type, "plot"))
        draw_headers = bool(_first(draw_headers, True))

        cfg_src = _first(xz_config, "")

        if image is None:
            raise ValueError("image is required")

        images_list = image if isinstance(image, list) else None

        def _explode(t):
            if isinstance(t, torch.Tensor):
                if t.ndim == 3:
                    return [t]
                if t.ndim == 4:
                    return [t[i] for i in range(int(t.shape[0]))]
            return []

        if images_list is not None:
            if len(images_list) == 0:
                raise ValueError("image list is empty")
            tensors: List[torch.Tensor] = []
            for it in images_list:
                tensors.extend(_explode(it))
        else:
            if not isinstance(image, torch.Tensor):
                raise TypeError("Expected IMAGE as torch.Tensor or list[torch.Tensor]")
            tensors = _explode(image)

        if len(tensors) == 0:
            raise ValueError("No images to process")

        batch = len(tensors)
        if batch == 1:
            return (tensors[0].unsqueeze(0), cfg_src)
        first_pil = _image_tensor_to_pil(tensors[0])
        w, h = first_pil.size

        header_height = int(round(h * (float(header_height_percent) / 100.0))) if draw_headers else 0
        cfg_list = _parse_xz_config(cfg_src)
        processed_pils: List[Image.Image] = []
        meta_list: List[Dict[str, str]] = []
        header_layout = {}
        for i in range(batch):
            pil = _image_tensor_to_pil(tensors[i])
            if pil.size != (w, h):
                pil = pil.resize((w, h), resample=Image.BILINEAR)

            x_text = ""
            z_text = ""
            meta = {"seed": "", "steps": "", "cfg": "", "denoise": "", "lora": "", "pos": "", "neg": ""}

            if i < len(cfg_list):
                v = cfg_list[i]
                if isinstance(v, dict):
                    xv = v.get("x_text", "")
                    zv = v.get("z_text", "")
                    x_text = "" if xv is None else str(xv)
                    z_text = "" if zv is None else str(zv)
                    meta = _build_meta(v)

            pimg = _add_header_and_text(pil, header_height, x_text, z_text, sting_trim, header_layout)
            processed_pils.append(pimg)
            meta_list.append(meta)

        if str(plot_type).lower() == "plot":
            gs = int(max(0, grid_spacing))
            out_w = (w * batch) + (gs * (batch - 1) if batch > 1 else 0)
            out_h = h + header_height
            output_plot = Image.new("RGB", (out_w, out_h), (0, 0, 0))
            for i, pimg in enumerate(processed_pils):
                x_off = i * (w + gs)
                output_plot.paste(pimg, (x_off, 0))

            out_t = _pil_to_image_tensor(output_plot).unsqueeze(0)
            return (out_t, cfg_src)

        out_batch = torch.stack([
            _pil_to_image_tensor(pimg) for i, pimg in enumerate(processed_pils)
        ], dim=0)
        return (out_batch, cfg_src)


NODE_CLASS_MAPPINGS = {
    "AK XZ Axis Plot": AKXZAxisPlot
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AK XZ Axis Plot": "AK XZ Axis Plot"
}