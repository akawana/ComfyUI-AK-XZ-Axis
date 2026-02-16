# AKXZKSampler.py
# A KSampler-like node that can accept either single values or lists on all main inputs.
# It normalizes all inputs to lists of the same length by padding with the last element.

import nodes


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
        # This should not happen when 'length' is computed as the max length,
        # but keep it safe and explicit.
        return lst[:length]
    last = lst[-1]
    return lst + [last] * (length - len(lst))


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
                "sampler_name": (nodes.KSampler.SAMPLERS,),
                "scheduler": (nodes.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "AK/XZ Axis"

    # Allow inputs to be lists. ComfyUI may still pass singletons; we normalize anyway.
    INPUT_IS_LIST = True
    # Return a list of latents so downstream nodes can map over them.
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

        # If everything is a single item, just do one render.
        if iterations <= 1:
            iterations = 1

        # Normalize all inputs to lists of the same length by padding with the last element.
        models = _pad_to_length(models, iterations)
        pos = _pad_to_length(pos, iterations)
        neg = _pad_to_length(neg, iterations)
        latents = _pad_to_length(latents, iterations)

        # KSampler node implementation
        ks = nodes.KSampler()

        out_latents = []
        base_seed = int(seed)

        for i in range(iterations):
            m = models[i]
            p = pos[i]
            n = neg[i]
            l = latents[i]

            # Use seed offset per item to avoid identical outputs when sampling multiple items
            item_seed = (base_seed + i) & 0xFFFFFFFF

            # nodes.KSampler.sample returns (latent,)
            (sampled_latent,) = ks.sample(
                m,
                p,
                n,
                l,
                item_seed,
                int(steps),
                float(cfg),
                sampler_name,
                scheduler,
                float(denoise),
            )

            out_latents.append(sampled_latent)

        return (out_latents,)


NODE_CLASS_MAPPINGS = {
    "AKXZKSampler": AKXZKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AKXZKSampler": "AK XZ KSampler (List Inputs)",
}
