# AKXZKSampler.py
# A KSampler-like node that can accept either single values or lists on all main inputs.
# It normalizes all main inputs to lists of the same length by padding with the last element.
# Note: when INPUT_IS_LIST=True, widget values may arrive wrapped in a list of length 1.

import nodes
import comfy.samplers


def _as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _pad_to_length(lst, length):
    if len(lst) == length:
        return lst
    if len(lst) == 0:
        raise ValueError("AKXZKSampler: cannot pad an empty list.")
    if len(lst) > length:
        return lst[:length]
    last = lst[-1]
    return lst + [last] * (length - len(lst))


def _scalar(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        return x[0]
    return x


class AKXZKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "AK"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
    ):
        models = _as_list(model)
        pos = _as_list(positive)
        neg = _as_list(negative)
        latents = _as_list(latent_image)

        # Determine iterations as the maximum input length.
        iterations = max(len(models), len(pos), len(neg), len(latents))
        if iterations <= 1:
            iterations = 1

        # Normalize all main inputs to lists of the same length by padding with the last element.
        models = _pad_to_length(models, iterations)
        pos = _pad_to_length(pos, iterations)
        neg = _pad_to_length(neg, iterations)
        latents = _pad_to_length(latents, iterations)

        # Normalize widget values (they may come wrapped in a list due to INPUT_IS_LIST).
        base_seed = int(_scalar(seed) or 0)
        steps_v = int(_scalar(steps) or 1)
        cfg_v = float(_scalar(cfg) or 0.0)
        sampler_name_v = _scalar(sampler_name)
        scheduler_v = _scalar(scheduler)
        denoise_v = float(_scalar(denoise) or 0.0)

        ks = nodes.KSampler()

        out_latents = []
        for i in range(iterations):
            m = models[i]
            p = pos[i]
            n = neg[i]
            l = latents[i]

            # Use seed offset per item to avoid identical outputs when sampling multiple items.
            item_seed = (base_seed + i) & 0xFFFFFFFF

            (sampled_latent,) = ks.sample(
                m,
                p,
                n,
                l,
                item_seed,
                steps_v,
                cfg_v,
                sampler_name_v,
                scheduler_v,
                denoise_v,
            )

            out_latents.append(sampled_latent)

        return (out_latents,)


NODE_CLASS_MAPPINGS = {
    "AKXZKSampler": AKXZKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AKXZKSampler": "AK XZ KSampler (List Inputs)",
}
