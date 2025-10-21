#!/usr/bin/env python
"""Qwen2.5-VL LMVR-style visual token extractor and attention visualizer.

Dependencies (suggested versions):
- transformers>=4.50.0
- torch>=2.2
- pillow
- numpy
- matplotlib
- tqdm
- (optional) qwen-vl-utils

Install example:
    pip install "transformers>=4.50" torch pillow numpy matplotlib tqdm qwen-vl-utils

This script loads Qwen2.5-VL-7B-Instruct (or other model variants that share the same
API), extracts the final-layer visual token hidden states, saves them to disk, and
produces token-level attention visualizations overlayed on the original image.
"""

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:  # Optional utility for Qwen vision inputs.
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    process_vision_info = None  # type: ignore


LOGGER = logging.getLogger("qwen25vl_vis_tokens")


@dataclass
class VisualTokenInfo:
    mask: torch.Tensor
    indices: torch.Tensor
    grid_h: int
    grid_w: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Qwen2.5-VL visual tokens and visualize attentions."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to image file.")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question or prompt for the multimodal input.",
    )
    parser.add_argument(
        "--target_word",
        type=str,
        default=None,
        help="Target word to visualize. Defaults to the last textual token if unset.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu'). Defaults to model's device.",
    )
    parser.add_argument(
        "--vis_layer",
        type=int,
        default=None,
        help="Layer index for attention visualization. -1 means last layer. Default: middle layer.",
    )
    parser.add_argument(
        "--use_last_layer_for_vis",
        action="store_true",
        help="Force using the final layer for visualization, overriding --vis_layer unless set.",
    )
    parser.add_argument(
        "--image_index",
        type=int,
        default=0,
        help="Index of the image to visualize if multiple images are provided.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Directory to save extracted features and visualizations.",
    )
    parser.add_argument(
        "--heat_alpha",
        type=float,
        default=0.5,
        help="Alpha value for overlaying the heatmap onto the original image.",
    )
    parser.add_argument(
        "--save_heatmap_raw",
        action="store_true",
        help="If set, also save the raw heatmap as a .npy file.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    return args


def load_model_and_processor(model_id: str, device: Optional[str] = None):
    LOGGER.info("Loading model %s", model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    if device is not None:
        model = model.to(device)
    LOGGER.info("Model loaded on device(s): %s", model.device)
    return model, processor


def build_messages(image_path: str, question: str) -> Tuple[List[Dict], Image.Image]:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return messages, image


def build_inputs(
    processor: AutoProcessor,
    messages: List[Dict],
    image_index: int = 0,
) -> Tuple[Dict[str, torch.Tensor], str]:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    images: Optional[Sequence[Image.Image]] = None
    if process_vision_info is not None:
        vision_infos = process_vision_info(messages)
        images = vision_infos["images"]
    if images is None:
        # Fallback: collect PIL images manually from messages.
        collected: List[Image.Image] = []
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                    collected.append(item["image"])
        images = collected
    if not images:
        raise ValueError("No images found in the provided messages.")
    if image_index >= len(images):
        raise IndexError(
            f"Requested image_index {image_index}, but only {len(images)} image(s) were provided."
        )
    selected_image = images[image_index]
    inputs = processor(
        text=[text],
        images=[selected_image],
        return_tensors="pt",
        return_dict=True,
    )
    return inputs, text


def move_inputs_to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def locate_visual_tokens(
    input_ids: torch.Tensor,
    config,
    extra_inputs: Dict[str, torch.Tensor],
) -> VisualTokenInfo:
    vision_start_token_id = getattr(config, "vision_start_token_id", None)
    vision_end_token_id = getattr(config, "vision_end_token_id", None)
    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Model config does not define image_token_id; cannot identify visual tokens.")
    ids = input_ids
    vis_mask = ids == image_token_id
    if vision_start_token_id is not None and vision_end_token_id is not None:
        start_positions = (ids == vision_start_token_id).nonzero(as_tuple=True)[0]
        end_positions = (ids == vision_end_token_id).nonzero(as_tuple=True)[0]
        if len(start_positions) > 0 and len(end_positions) > 0:
            start = start_positions[0]
            # End token is inclusive; mask tokens strictly between start and end.
            end_candidates = end_positions[end_positions > start]
            if len(end_candidates) == 0:
                LOGGER.warning("vision_end_token_id not found after vision_start_token_id; using all image tokens.")
            else:
                end = end_candidates[0]
                range_mask = torch.zeros_like(ids, dtype=torch.bool)
                range_mask[start:end] = True  # exclude the end token itself
                vis_mask = vis_mask & range_mask
    indices = vis_mask.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        raise ValueError("No visual tokens located in the input sequence.")
    grid_h, grid_w = infer_grid_shape(indices.numel(), extra_inputs)
    return VisualTokenInfo(mask=vis_mask, indices=indices, grid_h=grid_h, grid_w=grid_w)


def infer_grid_shape(num_visual_tokens: int, extra_inputs: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    grid_h = grid_w = int(math.sqrt(num_visual_tokens))
    if "image_grid_thw" in extra_inputs:
        grid = extra_inputs["image_grid_thw"]
        if isinstance(grid, torch.Tensor):
            grid_np = grid.cpu().numpy()
            if grid_np.ndim >= 3:
                # Expect shape [batch, num_images, 3]
                t, h, w = grid_np[0, 0]
                if t > 1:
                    LOGGER.warning(
                        "Temporal dimension t=%d detected; flattening to first frame for visualization.",
                        int(t),
                    )
                grid_h = int(h)
                grid_w = int(w)
            elif grid_np.ndim == 2:
                t, h, w = grid_np[0]
                grid_h = int(h)
                grid_w = int(w)
            else:
                LOGGER.warning(
                    "Unexpected image_grid_thw shape %s; falling back to sqrt heuristic.",
                    grid_np.shape,
                )
    if grid_h * grid_w != num_visual_tokens:
        LOGGER.warning(
            "Grid size %d x %d != number of visual tokens %d; using sqrt heuristic.",
            grid_h,
            grid_w,
            num_visual_tokens,
        )
        grid_h = grid_w = int(math.sqrt(num_visual_tokens))
        if grid_h * grid_w != num_visual_tokens:
            LOGGER.warning(
                "Non-square visual token grid (n=%d); using grid_h=%d, grid_w=%d with padding if needed.",
                num_visual_tokens,
                grid_h,
                grid_w,
            )
            # Determine width via ceiling division.
            grid_w = math.ceil(num_visual_tokens / grid_h)
    return grid_h, grid_w


def select_target_token(
    tokenizer,
    raw_text: str,
    input_ids: torch.Tensor,
    visual_mask: torch.Tensor,
    target_word: Optional[str],
) -> int:
    ids = input_ids.tolist()
    token_ids = ids
    if target_word:
        target_ids = tokenizer.encode(target_word, add_special_tokens=False)
        if target_ids:
            match_idx = locate_subsequence(token_ids, target_ids)
            if match_idx is not None:
                q_idx = match_idx + len(target_ids) - 1
                LOGGER.info(
                    "Target word '%s' matched token ids %s at position %d.",
                    target_word,
                    target_ids,
                    q_idx,
                )
                return q_idx
            LOGGER.warning("Target word '%s' not found in tokenized sequence; falling back.", target_word)
        else:
            LOGGER.warning("Tokenizer produced no ids for target word '%s'; falling back.", target_word)
    # Fallback: choose the last text token outside visual tokens (and not padding).
    special_token_ids = {
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        getattr(tokenizer, "bos_token_id", None),
    }
    for idx in range(len(token_ids) - 1, -1, -1):
        if visual_mask[idx]:
            continue
        tok_id = token_ids[idx]
        if tok_id is None or tok_id in special_token_ids:
            continue
        LOGGER.info("Fallback target token index %d (id=%d).", idx, tok_id)
        return idx
    raise ValueError("Unable to find a textual token to visualize.")


def locate_subsequence(sequence: Sequence[int], subseq: Sequence[int]) -> Optional[int]:
    if not subseq:
        return None
    sub_len = len(subseq)
    for start in range(0, len(sequence) - sub_len + 1):
        if list(sequence[start : start + sub_len]) == list(subseq):
            return start
    return None


def extract_last_layer_visual_features(
    hidden_states: torch.Tensor,
    vis_info: VisualTokenInfo,
) -> torch.Tensor:
    vis_features = hidden_states[vis_info.mask]
    if vis_features.dim() != 2:
        vis_features = vis_features.view(vis_info.indices.numel(), -1)
    return vis_features


def compute_attention_heatmap(
    attentions: Sequence[torch.Tensor],
    layer_index: int,
    query_index: int,
    vis_info: VisualTokenInfo,
) -> torch.Tensor:
    num_layers = len(attentions)
    if layer_index < 0:
        layer_index = num_layers + layer_index
    if not (0 <= layer_index < num_layers):
        raise IndexError(
            f"Layer index {layer_index} is out of range for {num_layers} layers."
        )
    attn_layer = attentions[layer_index][0]  # [heads, q_len, k_len]
    attn_avg = attn_layer.mean(dim=0)  # [q_len, k_len]
    attn_vector = attn_avg[query_index]  # [k_len]
    heat_values = attn_vector[vis_info.mask]
    if heat_values.numel() != vis_info.indices.numel():
        raise RuntimeError("Mismatch between attention visual tokens and located indices.")
    expected = vis_info.grid_h * vis_info.grid_w
    if expected == heat_values.numel():
        heatmap = heat_values.view(vis_info.grid_h, vis_info.grid_w)
    else:
        LOGGER.warning(
            "Product of grid_h (%d) and grid_w (%d) != number of visual tokens (%d); reshaping to 1xN.",
            vis_info.grid_h,
            vis_info.grid_w,
            heat_values.numel(),
        )
        heatmap = heat_values.view(1, -1)
    return heatmap


def normalize_heatmap(heatmap: torch.Tensor) -> np.ndarray:
    heat = heatmap.detach().cpu().float().numpy()
    heat = np.maximum(heat, 0.0)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def overlay_heatmap_on_image(
    image_path: str,
    heatmap: np.ndarray,
    out_path: str,
    alpha: float = 0.5,
):
    base_image = Image.open(image_path).convert("RGB")
    heat_image = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    heat_resized = heat_image.resize(base_image.size, resample=Image.BILINEAR)
    heat_arr = np.asarray(heat_resized).astype(np.float32) / 255.0
    base_arr = np.asarray(base_image).astype(np.float32) / 255.0
    import matplotlib.cm as cm

    colormap = cm.get_cmap("jet")
    colored = colormap(heat_arr)[..., :3]
    overlay = (1 - alpha) * base_arr + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def save_visual_features(
    features: torch.Tensor,
    vis_info: VisualTokenInfo,
    out_dir: str,
    base_filename: str,
):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{base_filename}_vis_tokens.npz")
    np.savez(
        npz_path,
        visual_tokens=features.detach().cpu().numpy(),
        grid_h=vis_info.grid_h,
        grid_w=vis_info.grid_w,
        visual_indices=vis_info.indices.detach().cpu().numpy(),
    )
    LOGGER.info("Saved visual token features to %s", npz_path)


def main(args: argparse.Namespace) -> None:
    model, processor = load_model_and_processor(args.model_id, args.device)
    messages, _ = build_messages(args.image, args.question)
    inputs, raw_text = build_inputs(processor, messages, args.image_index)
    inputs_on_device = move_inputs_to_device(inputs, model.device)

    LOGGER.info("Running forward pass with output_hidden_states=True, output_attentions=True")
    with torch.no_grad():
        outputs = model(
            **inputs_on_device,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    last_hidden = hidden_states[-1][0]  # [seq_len, hidden]
    input_ids = inputs_on_device["input_ids"][0]

    vis_info = locate_visual_tokens(input_ids, model.config, inputs_on_device)
    LOGGER.info(
        "Located %d visual tokens with grid %dx%d.",
        vis_info.indices.numel(),
        vis_info.grid_h,
        vis_info.grid_w,
    )

    vis_features = extract_last_layer_visual_features(last_hidden, vis_info)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    save_visual_features(vis_features, vis_info, args.out_dir, base_name)

    num_layers = len(attentions)
    if num_layers == 0:
        raise RuntimeError("Model did not return attention tensors; ensure output_attentions=True.")
    if args.use_last_layer_for_vis:
        chosen_layer = num_layers - 1
    elif args.vis_layer is not None:
        chosen_layer = args.vis_layer if args.vis_layer >= 0 else num_layers + args.vis_layer
    else:
        chosen_layer = num_layers // 2
    chosen_layer = max(0, min(num_layers - 1, chosen_layer))
    LOGGER.info("Using layer %d (0-indexed) for attention visualization.", chosen_layer)

    tokenizer = processor.tokenizer
    query_index = select_target_token(tokenizer, raw_text, input_ids, vis_info.mask, args.target_word)
    LOGGER.info("Selected query token index %d.", query_index)

    heatmap_tensor = compute_attention_heatmap(attentions, chosen_layer, query_index, vis_info)
    heatmap = normalize_heatmap(heatmap_tensor)

    overlay_path = os.path.join(args.out_dir, f"{base_name}_overlay.png")
    overlay_heatmap_on_image(args.image, heatmap, overlay_path, alpha=args.heat_alpha)
    LOGGER.info("Saved attention overlay to %s", overlay_path)

    if args.save_heatmap_raw:
        raw_path = os.path.join(args.out_dir, f"{base_name}_heatmap.npy")
        np.save(raw_path, heatmap)
        LOGGER.info("Saved raw heatmap to %s", raw_path)

    flat_heat = heatmap.reshape(-1)
    if flat_heat.size > 0:
        topk = min(5, flat_heat.size)
        top_indices = np.argpartition(-flat_heat, list(range(topk)))[:topk]
        top_values = flat_heat[top_indices]
        LOGGER.info(
            "Top-%d attention positions (index:value): %s",
            topk,
            sorted(
                [
                    (int(idx), float(val))
                    for idx, val in zip(top_indices, top_values)
                ],
                key=lambda x: x[1],
                reverse=True,
            ),
        )

    summary = {
        "num_visual_tokens": int(vis_info.indices.numel()),
        "grid_h": vis_info.grid_h,
        "grid_w": vis_info.grid_w,
        "query_index": int(query_index),
        "layer_index": int(chosen_layer),
    }
    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # Example usage (requires valid image path):
    # python qwen25vl_vis_tokens.py \
    #   --image path/to/example.jpg \
    #   --question "Describe the image in one sentence. Focus on the cat." \
    #   --target_word "cat" \
    #   --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
    #   --device "cuda" \
    #   --vis_layer -1
