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

local_dir = "/home/host/qwen2.5-vl/"  # 你的本地路径

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
        "--temp",
        type=float,
        default=0.5,
        help="Temperature for softmax normalization over visual tokens.",
    )
    parser.add_argument(
        "--head_pool",
        type=str,
        choices=["mean", "max"],
        default="max",
        help="Pooling strategy for aggregating attention heads.",
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
    
    # 禁用 Flash Attention 以支持 output_attentions
    # 方法1: 通过 attn_implementation 参数
    if device is not None and device.startswith("cuda") and "," not in device:
        # 单GPU模式
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype="auto",
            attn_implementation="eager"  # 使用标准 attention 而不是 flash attention
        )
        model = model.to(device)
    else:
        # 多GPU或自动模式
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype="auto", 
            device_map="auto",
            attn_implementation="eager"  # 使用标准 attention
        )
    
    processor = AutoProcessor.from_pretrained(model_id)
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
    # 注释掉有问题的代码，直接使用 fallback 逻辑
    # if process_vision_info is not None:
    #     vision_infos = process_vision_info(messages)
    #     images = vision_infos["images"]
    
    # 直接从 messages 中提取图片
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
    image_positions = (ids == image_token_id).nonzero(as_tuple=True)[0]
    if image_positions.numel() == 0:
        raise ValueError("No visual tokens located in the input sequence.")

    segments: List[torch.Tensor] = []
    if vision_start_token_id is not None and vision_end_token_id is not None:
        start_positions = (ids == vision_start_token_id).nonzero(as_tuple=True)[0].tolist()
        end_positions = (ids == vision_end_token_id).nonzero(as_tuple=True)[0].tolist()
        remaining_end_positions = end_positions.copy()
        for start in start_positions:
            end_candidates = [e for e in remaining_end_positions if e > start]
            if not end_candidates:
                LOGGER.warning(
                    "vision_end_token_id not found after vision_start_token_id at position %d; skipping.",
                    start,
                )
                continue
            end = end_candidates[0]
            segment = image_positions[(image_positions > start) & (image_positions < end)]
            if segment.numel() > 0:
                segments.append(segment)
            remaining_end_positions = [e for e in remaining_end_positions if e > end]
        if not segments:
            LOGGER.warning(
                "No image tokens located between vision_start_token_id and vision_end_token_id; falling back to contiguous image tokens.",
            )

    if not segments:
        # Fallback: split by contiguous spans of image tokens.
        positions_list = image_positions.tolist()
        current_segment: List[int] = [positions_list[0]]
        contiguous_segments: List[List[int]] = []
        for pos in positions_list[1:]:
            if pos == current_segment[-1] + 1:
                current_segment.append(pos)
            else:
                contiguous_segments.append(current_segment)
                current_segment = [pos]
        contiguous_segments.append(current_segment)
        if len(contiguous_segments) > 1:
            LOGGER.warning(
                "Multiple visual token segments detected; using the first segment only. TODO: support multi-segment vision inputs.",
            )
        chosen = torch.tensor(contiguous_segments[0], device=ids.device, dtype=torch.long)
    else:
        if len(segments) > 1:
            LOGGER.warning(
                "Multiple <vision> segments detected; using the first segment only. TODO: support multi-segment vision inputs.",
            )
        chosen = segments[0]

    vis_mask = torch.zeros_like(ids, dtype=torch.bool)
    vis_mask[chosen] = True
    indices = vis_mask.nonzero(as_tuple=True)[0]
    grid_h, grid_w = infer_grid_shape(indices.numel(), extra_inputs)
    return VisualTokenInfo(mask=vis_mask, indices=indices, grid_h=grid_h, grid_w=grid_w)


def infer_grid_shape(num_visual_tokens: int, extra_inputs: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "image_grid_thw" in extra_inputs:
        thw = extra_inputs["image_grid_thw"]
        thw = thw[0]
        if thw.ndim == 1:
            t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
        else:
            t, h, w = int(thw[-3]), int(thw[-2]), int(thw[-1])
        if h * w == num_visual_tokens:
            return h, w
        LOGGER.warning(
            "image_grid_thw says %dx%d (t=%d) but N_vis=%d; falling back.",
            h,
            w,
            t,
            num_visual_tokens,
        )
    grid_h = max(1, int(math.sqrt(num_visual_tokens)))
    grid_w = int(math.ceil(num_visual_tokens / grid_h))
    LOGGER.info(
        "Inferred grid shape: %dx%d=%d for %d visual tokens",
        grid_h,
        grid_w,
        grid_h * grid_w,
        num_visual_tokens,
    )
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
        # 尝试多种编码方式来匹配目标词
        target_variants = [
            target_word,
            target_word.lower(),
            target_word.upper(),
            target_word.capitalize(),
            " " + target_word,  # 带前导空格
            target_word + " ",  # 带后导空格
        ]
        
        matched = False
        for variant in target_variants:
            target_ids = tokenizer.encode(variant, add_special_tokens=False)
            if target_ids:
                match_idx = locate_subsequence(token_ids, target_ids)
                if match_idx is not None:
                    q_idx = match_idx + len(target_ids) - 1
                    LOGGER.info(
                        "Target word '%s' (variant: '%s') matched token ids %s at position %d.",
                        target_word,
                        variant,
                        target_ids,
                        q_idx,
                    )
                    return q_idx
        
        # 如果上述方法都失败，尝试在解码后的文本中查找
        LOGGER.info("Attempting to find '%s' by decoding tokens...", target_word)
        for i, tok_id in enumerate(token_ids):
            if visual_mask[i]:
                continue
            decoded = tokenizer.decode([tok_id], skip_special_tokens=True).strip().lower()
            if target_word.lower() in decoded or decoded in target_word.lower():
                LOGGER.info(
                    "Target word '%s' found via decoding at position %d (decoded: '%s', id=%d).",
                    target_word,
                    i,
                    decoded,
                    tok_id,
                )
                return i
        
        LOGGER.warning("Target word '%s' not found in tokenized sequence; falling back.", target_word)
    
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
        decoded = tokenizer.decode([tok_id], skip_special_tokens=True)
        LOGGER.info(
            "Fallback target token index %d (id=%d, decoded='%s').",
            idx,
            tok_id,
            decoded,
        )
        return idx
    raise ValueError("Unable to find a textual token to visualize.")


def locate_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def find_token_in_generated(tokenizer, gen_ids: torch.Tensor, target_word: str | None) -> int | None:
    """返回生成段中 target_word 的最后一个子词的相对索引；找不到则返回最后一个非特殊 token；都无则 None。"""
    if gen_ids.numel() == 0:
        return None
    seq = gen_ids[0].tolist()
    if target_word:
        for v in (target_word, target_word.lower(), target_word.capitalize()):
            sub = tokenizer.encode(v, add_special_tokens=False)
            if sub:
                pos = locate_subsequence(seq, sub)
                if pos is not None:
                    return pos + len(sub) - 1
    specials = {
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        getattr(tokenizer, "bos_token_id", -1),
    }
    specials = {int(s) for s in specials if s is not None and s != -1}
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] not in specials:
            return i
    return None


def pick_most_focused_generated_token(
    attentions,
    layer_index: int,
    gen_abs_start: int,
    full_seq_len: int,
    vis_mask_full: torch.Tensor,
    head_pool: str = "max",
) -> int | None:
    """在“完整序列”的生成段中，选择对视觉子集最聚焦的 token（返回绝对索引）。"""
    attn_layer = attentions[layer_index][0].to(torch.float32)  # [H,Q,K]
    A = attn_layer.max(0).values if head_pool == "max" else attn_layer.mean(0)  # [Q,K]
    best_idx, best_score = None, -1.0
    for q_abs in range(gen_abs_start, full_seq_len):
        a_q = A[q_abs]
        hv = a_q[vis_mask_full]
        if hv.numel() == 0:
            continue
        p = torch.softmax(hv, dim=-1)
        if p.numel() == 1:
            score = 1.0
        else:
            score = 1.0 - (-(p * (p + 1e-12).log()).sum() / math.log(p.numel())).item()
        if score > best_score:
            best_score, best_idx = score, q_abs
    return best_idx


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
    temp: float = 0.5,
    head_pool: str = "max",
) -> torch.Tensor:
    num_layers = len(attentions)
    if layer_index < 0:
        layer_index += num_layers
    if not (0 <= layer_index < num_layers):
        raise IndexError(f"Layer index {layer_index} is out of range for {num_layers} layers.")
    attn_layer = attentions[layer_index]
    if attn_layer is None:
        raise RuntimeError(f"Attention at layer {layer_index} is None; cannot compute heatmap.")
    attn_layer = attn_layer[0].to(torch.float32)  # [H, Q, K]
    A = attn_layer.max(0).values if head_pool == "max" else attn_layer.mean(0)  # [Q, K]
    if query_index >= A.shape[0]:
        raise IndexError(
            f"Query index {query_index} is out of range for sequence length {A.shape[0]}."
        )
    if vis_info.mask.shape[0] != A.shape[1]:
        raise ValueError("Visual token mask length does not match attention key dimension.")
    a_q = A[query_index]
    hv = a_q[vis_info.mask]
    if hv.numel() == 0:
        raise RuntimeError("No visual tokens selected by vis_info.mask.")
    hv = torch.softmax(hv / temp, dim=-1)
    expected = vis_info.grid_h * vis_info.grid_w
    if expected == hv.numel():
        return hv.view(vis_info.grid_h, vis_info.grid_w)
    LOGGER.warning(
        "grid_h*grid_w (%d) != N_vis (%d); fallback to 1xN.",
        expected,
        hv.numel(),
    )
    return hv.view(1, -1)


def normalize_heatmap(heatmap: torch.Tensor) -> np.ndarray:
    heat = heatmap.detach().cpu().float().numpy()
    heat = np.maximum(heat, 0.0)
    hmin, hmax = float(heat.min()), float(heat.max())
    if hmax > hmin:
        heat = (heat - hmin) / (hmax - hmin + 1e-6)
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
    
    # 转换为 float32 以兼容 numpy（BFloat16 不被直接支持）
    features_np = features.detach().cpu().float().numpy()
    indices_np = vis_info.indices.detach().cpu().numpy()
    
    np.savez(
        npz_path,
        visual_tokens=features_np,
        grid_h=vis_info.grid_h,
        grid_w=vis_info.grid_w,
        visual_indices=indices_np,
    )
    LOGGER.info("Saved visual token features to %s", npz_path)


def main(args: argparse.Namespace) -> None:
    model, processor = load_model_and_processor(args.model_id, args.device)
    messages, _ = build_messages(args.image, args.question)
    inputs, raw_text = build_inputs(processor, messages, args.image_index)
    inputs_on_device = move_inputs_to_device(inputs, model.device)

    LOGGER.info(
        "Running initial forward pass with output_hidden_states=True for visual token extraction."
    )
    with torch.no_grad():
        outputs = model(
            **inputs_on_device,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
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

    # === Generation followed by full-sequence attention extraction ===
    original_seq_len = int(inputs_on_device["input_ids"].shape[1])
    with torch.no_grad():
        gen_out = model.generate(
            **inputs_on_device,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
        )
    gen_ids = gen_out[:, inputs_on_device["input_ids"].shape[1] :]
    gen_len = int(gen_ids.shape[1])
    LOGGER.info(
        "Generated %d new token(s) from the model (prompt length %d).",
        gen_len,
        original_seq_len,
    )
    LOGGER.info(
        "Generated text: %s",
        processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True),
    )

    full_ids = torch.cat([inputs_on_device["input_ids"], gen_ids], dim=1)
    if "attention_mask" in inputs_on_device:
        attn_dtype = inputs_on_device["attention_mask"].dtype
    else:
        attn_dtype = torch.long
    full_attn_mask = torch.ones_like(
        full_ids,
        dtype=attn_dtype,
        device=full_ids.device,
    )
    full_inputs = dict(inputs_on_device)
    full_inputs["input_ids"] = full_ids
    full_inputs["attention_mask"] = full_attn_mask

    LOGGER.info(
        "Combined sequence length: %d (prompt %d + generated %d).",
        full_ids.shape[1],
        original_seq_len,
        gen_len,
    )
    LOGGER.info(
        "Running second forward pass for attention extraction with output_attentions=True."
    )
    with torch.no_grad():
        outputs_full = model(
            **full_inputs,
            output_attentions=True,
            use_cache=False,
        )
    attentions = outputs_full.attentions

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
    LOGGER.info("Using layer %d/%d for visualization.", chosen_layer, num_layers)

    tokenizer = processor.tokenizer

    LOGGER.debug("Raw text: %s", raw_text[:200] + "..." if len(raw_text) > 200 else raw_text)
    LOGGER.debug("Input IDs shape: %s", input_ids.shape)
    LOGGER.debug("Number of visual tokens: %d", vis_info.mask.sum().item())

    if args.target_word:
        test_encoding = tokenizer.encode(args.target_word, add_special_tokens=False)
        LOGGER.debug("Target word '%s' encodes to: %s", args.target_word, test_encoding)

    vis_mask_full = torch.zeros(full_ids.shape[1], dtype=torch.bool, device=full_ids.device)
    vis_mask_full[:original_seq_len] = vis_info.mask
    vis_info_for_attn = VisualTokenInfo(
        mask=vis_mask_full,
        indices=vis_mask_full.nonzero(as_tuple=True)[0],
        grid_h=vis_info.grid_h,
        grid_w=vis_info.grid_w,
    )

    rel = find_token_in_generated(tokenizer, gen_ids, args.target_word)
    if rel is not None:
        query_index = original_seq_len + int(rel)
        LOGGER.info(
            "Use GENERATED '%s' token at abs idx %d (rel %d).",
            args.target_word,
            query_index,
            rel,
        )
    else:
        auto_idx = pick_most_focused_generated_token(
            attentions,
            chosen_layer,
            original_seq_len,
            int(full_ids.shape[1]),
            vis_mask_full,
            head_pool=args.head_pool,
        )
        if auto_idx is not None:
            query_index = int(auto_idx)
            LOGGER.info("Auto-picked most focused generated token at abs idx %d.", query_index)
        else:
            query_index = select_target_token(
                tokenizer,
                raw_text,
                input_ids,
                vis_info.mask,
                args.target_word,
            )
            LOGGER.info("Fallback to INPUT segment, abs idx %d.", query_index)

    last_vis = int(vis_info.indices.max().item())
    LOGGER.info("Last visual token index: %d.", last_vis)
    assert query_index > last_vis, (
        f"query_index({query_index}) must be AFTER last visual token ({last_vis})."
    )

    heatmap_tensor = compute_attention_heatmap(
        attentions=attentions,
        layer_index=chosen_layer,
        query_index=query_index,
        vis_info=vis_info_for_attn,
        temp=args.temp,
        head_pool=args.head_pool,
    )
    heatmap = normalize_heatmap(heatmap_tensor)

    overlay_path = os.path.join(args.out_dir, f"{base_name}_overlay.png")
    overlay_heatmap_on_image(args.image, heatmap, overlay_path, alpha=args.heat_alpha)
    LOGGER.info("Saved attention overlay to %s", overlay_path)

    if args.save_heatmap_raw:
        raw_path = os.path.join(args.out_dir, f"{base_name}_heatmap.npy")
        # heatmap 已经是 numpy array，应该没问题
        # 但如果有问题，可以确保是 float32
        np.save(raw_path, heatmap.astype(np.float32))
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
        "generated_tokens": int(gen_len),
        "query_index": int(query_index),
        "layer_index": int(chosen_layer),
    }
    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)

'''
Example usage (requires valid image path):
python qwen25vl_vis_tokens.py \
    --image path/to/example.jpg \
    --question "Describe the image in one sentence. Focus on the cat." \
    --target_word "cat" \
    --model_id "/home/host/qwen2.5-vl/" \
    --device "cuda:0,1,2,3" \
    --vis_layer -1
'''
