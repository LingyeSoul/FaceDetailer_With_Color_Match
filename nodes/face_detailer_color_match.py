import copy
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import nodes


def _resolve_face_detailer_class():
    face_detailer_class = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("FaceDetailer")
    if face_detailer_class is None:
        raise Exception(
            "FaceDetailer 未找到。请先安装并启用 ComfyUI-Impact-Pack，然后重启 ComfyUI。"
        )
    return face_detailer_class


def _get_mask_bbox(mask_2d, threshold=1e-4):
    active = mask_2d > threshold
    if not torch.any(active):
        return None

    coords = torch.nonzero(active, as_tuple=False)
    y0 = int(coords[:, 0].min().item())
    y1 = int(coords[:, 0].max().item()) + 1
    x0 = int(coords[:, 1].min().item())
    x1 = int(coords[:, 1].max().item()) + 1
    return y0, y1, x0, x1


def _color_match_batch(image_ref, image_target, method, strength=1.0, multithread=True, mask=None):
    try:
        from color_matcher import ColorMatcher
    except Exception as exc:
        raise Exception(
            "无法导入 color-matcher，请安装依赖：pip install color-matcher"
        ) from exc

    image_ref = image_ref.cpu()
    image_target = image_target.cpu()
    if mask is not None:
        mask = _align_mask_to_image(mask.cpu(), image_target)

    batch_size = image_target.size(0)

    if image_ref.size(0) not in (1, batch_size):
        raise Exception("参考图 batch 大小必须为 1 或与目标图一致")

    if mask is not None and mask.size(0) not in (1, batch_size):
        raise Exception("mask batch 大小必须为 1 或与目标图一致")

    def process(i):
        cm = ColorMatcher()
        image_target_i = image_target[0] if image_target.size(0) == 1 else image_target[i]
        image_ref_i = image_ref[0] if image_ref.size(0) == 1 else image_ref[i]

        if mask is None:
            image_target_np_i = image_target_i.numpy().copy()
            image_ref_np_i = image_ref_i.numpy().copy()
            try:
                image_result = cm.transfer(src=image_target_np_i, ref=image_ref_np_i, method=method)
                image_result = image_target_np_i + strength * (image_result - image_target_np_i)
                return torch.from_numpy(image_result)
            except Exception:
                return image_target_i

        mask_i = mask[0] if mask.size(0) == 1 else mask[i]
        mask_2d = mask_i[..., 0]
        bbox = _get_mask_bbox(mask_2d)
        if bbox is None:
            return image_target_i

        y0, y1, x0, x1 = bbox
        image_target_crop = image_target_i[y0:y1, x0:x1, :]
        image_ref_crop = image_ref_i[y0:y1, x0:x1, :]
        mask_crop = mask_i[y0:y1, x0:x1, :]

        image_target_np_i = image_target_crop.numpy().copy()
        image_ref_np_i = image_ref_crop.numpy().copy()
        try:
            image_result = cm.transfer(src=image_target_np_i, ref=image_ref_np_i, method=method)
            image_result = image_target_np_i + strength * (image_result - image_target_np_i)
            image_result = torch.from_numpy(image_result).to(torch.float32)
        except Exception:
            image_result = image_target_crop

        blended = image_target_i.clone()
        blended_crop = image_target_crop * (1.0 - mask_crop) + image_result * mask_crop
        blended[y0:y1, x0:x1, :] = blended_crop
        return blended

    if multithread and batch_size > 1:
        max_threads = min(os.cpu_count() or 1, batch_size)
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            out = list(executor.map(process, range(batch_size)))
    else:
        out = [process(i) for i in range(batch_size)]

    out = torch.stack(out, dim=0).to(torch.float32)
    out.clamp_(0, 1)
    return out


def _align_mask_to_image(mask, image):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    if mask.dim() == 3:
        mask = mask.unsqueeze(-1)
    elif mask.dim() == 4:
        if mask.shape[1] == 1 and mask.shape[-1] != 1:
            mask = mask.permute(0, 2, 3, 1)
        elif mask.shape[-1] != 1:
            mask = mask.mean(dim=-1, keepdim=True)
    else:
        raise Exception(f"不支持的 mask 维度: {mask.shape}")

    if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
        mask = F.interpolate(mask.permute(0, 3, 1, 2), size=(image.shape[1], image.shape[2]), mode="nearest")
        mask = mask.permute(0, 2, 3, 1)

    mask = mask.to(device=image.device, dtype=image.dtype)
    return mask.clamp(0.0, 1.0)


class FaceDetailerColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        face_detailer_class = _resolve_face_detailer_class()
        base = copy.deepcopy(face_detailer_class.INPUT_TYPES())

        optional = base.setdefault("optional", {})
        optional.update(
            {
                "apply_color_match": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "color_match_method": (
                    ["mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"],
                    {"default": "mkl"},
                ),
                "color_match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "color_match_multithread": ("BOOLEAN", {"default": True}),
            }
        )
        return base

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Simple"
    DESCRIPTION = (
        "基于 FaceDetailer 完整流程，完成遮罩区域重绘后，\n"
        "仅对遮罩对应区域做 Color Match，并按同一遮罩区域回贴。"
    )

    def doit(
        self,
        image,
        model,
        clip,
        vae,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        denoise,
        feather,
        noise_mask,
        force_inpaint,
        bbox_threshold,
        bbox_dilation,
        bbox_crop_factor,
        sam_detection_hint,
        sam_dilation,
        sam_threshold,
        sam_bbox_expansion,
        sam_mask_hint_threshold,
        sam_mask_hint_use_negative,
        drop_size,
        bbox_detector,
        wildcard,
        cycle=1,
        sam_model_opt=None,
        segm_detector_opt=None,
        detailer_hook=None,
        inpaint_model=False,
        noise_mask_feather=0,
        scheduler_func_opt=None,
        tiled_encode=False,
        tiled_decode=False,
        apply_color_match=True,
        color_match_method="mkl",
        color_match_strength=1.0,
        color_match_multithread=True,
    ):
        face_detailer_class = _resolve_face_detailer_class()
        base_node = face_detailer_class()

        result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images = base_node.doit(
            image,
            model,
            clip,
            vae,
            guide_size,
            guide_size_for,
            max_size,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            denoise,
            feather,
            noise_mask,
            force_inpaint,
            bbox_threshold,
            bbox_dilation,
            bbox_crop_factor,
            sam_detection_hint,
            sam_dilation,
            sam_threshold,
            sam_bbox_expansion,
            sam_mask_hint_threshold,
            sam_mask_hint_use_negative,
            drop_size,
            bbox_detector,
            wildcard,
            cycle=cycle,
            sam_model_opt=sam_model_opt,
            segm_detector_opt=segm_detector_opt,
            detailer_hook=detailer_hook,
            inpaint_model=inpaint_model,
            noise_mask_feather=noise_mask_feather,
            scheduler_func_opt=scheduler_func_opt,
            tiled_encode=tiled_encode,
            tiled_decode=tiled_decode,
        )

        if apply_color_match:
            matched = _color_match_batch(
                image_ref=image,
                image_target=result_img,
                method=color_match_method,
                strength=color_match_strength,
                multithread=color_match_multithread,
                mask=result_mask,
            )
            matched = matched.to(device=result_img.device, dtype=result_img.dtype)
            result_img = matched.clamp(0.0, 1.0)

        return (
            result_img,
            result_cropped_enhanced,
            result_cropped_enhanced_alpha,
            result_mask,
            pipe,
            result_cnet_images,
        )


NODE_CLASS_MAPPINGS = {
    "FaceDetailerColorMatch": FaceDetailerColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetailerColorMatch": "FaceDetailer + Color Match",
}
