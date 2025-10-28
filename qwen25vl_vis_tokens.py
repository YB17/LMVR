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
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import PhrasalConstraint

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
    grid_thw: Optional[List[Tuple[int, int, int]]] = None


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
        default=0.7,
        help="Temperature for softmax normalization over visual tokens.",
    )
    parser.add_argument(
        "--head_pool",
        type=str,
        choices=["mean", "max", "weighted"],
        default="weighted",
        help="Pooling strategy for aggregating attention heads.",
    )
    parser.add_argument(
        "--visual_ratio_gate",
        type=float,
        default=0.12,
        help="Minimum visual attention ratio required before applying subset softmax.",
    )
    parser.add_argument(
        "--use_constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enforce target phrase via constrained decoding (default: True).",
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
        "--dump_gen_tokens",
        action="store_true",
        help="Dump generated token pieces for debugging.",
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


def debug_dump_generated_tokens(tokenizer, gen_ids):
    toks = tokenizer.convert_ids_to_tokens(gen_ids[0].tolist())
    lines = []
    for i, t in enumerate(toks):
        piece = tokenizer.decode(gen_ids[0, i : i + 1], skip_special_tokens=True)
        lines.append(f"{i:>3}: {t!r} -> {piece!r}")
    LOGGER.info("GEN TOKENS:\n%s", "\n".join(lines))


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.lower())
    s = s.replace("Ġ", "").replace("▁", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w]", "", s)
    return s


def find_target_by_tokens(tokenizer, gen_ids, target_phrase: str):
    """在生成段中用滑窗拼接查找 target（支持短语）。返回 (相对末索引, 实际匹配文本) 或 (None, None)。"""

    if gen_ids is None or gen_ids.numel() == 0 or not target_phrase:
        return None, None
    target_norm = _norm_text(target_phrase)

    ids = gen_ids[0].tolist()
    pieces = [
        tokenizer.decode(gen_ids[0, i : i + 1], skip_special_tokens=True)
        for i in range(len(ids))
    ]
    norm = [_norm_text(p) for p in pieces]

    for i in range(len(norm)):
        buf = ""
        last = i
        for j in range(i, len(norm)):
            buf += norm[j]
            last = j
            if buf == target_norm:
                matched = "".join(pieces[i : last + 1])
                return last, matched
            if not target_norm.startswith(buf):
                break
    return None, None


def generate_with_constraints(model, tokenizer, inputs_on_device, phrase: str):
    ids = tokenizer.encode(phrase, add_special_tokens=False) if phrase else []
    constraints = [PhrasalConstraint(ids)] if ids else None
    bad = tokenizer.encode("addCriterion", add_special_tokens=False)
    kwargs = dict(
        **inputs_on_device,
        max_new_tokens=96,
        do_sample=False,
        num_beams=6,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=0.8,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if constraints:
        kwargs["constraints"] = constraints
    elif ids:
        kwargs["force_words_ids"] = [ids]
    if bad:
        kwargs["bad_words_ids"] = [bad]
    try:
        return model.generate(**kwargs)
    except Exception as exc:
        if "constraints" in kwargs and ids:
            LOGGER.warning(
                "Constraint decoding failed with constraints (%s); retrying with force_words_ids.",
                exc,
            )
            kwargs.pop("constraints", None)
            kwargs["force_words_ids"] = [ids]
            return model.generate(**kwargs)
        raise


def pooled_over_heads_fullK(attn_layer, query_index, vis_mask, head_pool="weighted"):
    A_q = attn_layer[:, query_index, :]
    if head_pool == "weighted":
        mass_all = A_q.sum(dim=-1).clamp_min(1e-6)
        mass_vis = A_q[:, vis_mask].sum(dim=-1)
        w = mass_vis / mass_all
        w = (w / w.sum().clamp_min(1e-6)).unsqueeze(-1)
        a_q = (A_q * w).sum(dim=0)
    elif head_pool == "mean":
        a_q = A_q.mean(dim=0)
    elif head_pool == "max":
        a_q = A_q.max(dim=0).values
    else:
        raise ValueError("head_pool must be 'weighted'|'mean'|'max'")
    return a_q


def pick_most_focused_generated_token(
    attentions,
    layer_index: int,
    gen_abs_start: int,
    full_seq_len: int,
    vis_mask_full: torch.Tensor,
    head_pool: str = "weighted",
    tokenizer=None,
    full_ids=None,
    gamma: float = 2.0,
    min_visual_ratio: float = 0.12,
    temp: float = 0.7,
):
    attn = attentions[layer_index][0].to(torch.float32)
    best = (None, None, -1.0, 0.0, 0.0)

    for q_abs in range(gen_abs_start, full_seq_len):
        A_q = attn[:, q_abs, :]
        mass_all_h = A_q.sum(dim=-1).clamp_min(1e-6)
        mass_vis_h = A_q[:, vis_mask_full].sum(dim=-1)

        if head_pool == "weighted":
            w = mass_vis_h / mass_all_h
            w = (w / w.sum().clamp_min(1e-6)).unsqueeze(-1)
            a_q = (A_q * w).sum(dim=0)
        elif head_pool == "mean":
            a_q = A_q.mean(dim=0)
        else:
            a_q = A_q.max(dim=0).values

        mass_all = float(a_q.sum().clamp_min(1e-6))
        hv_raw = a_q[vis_mask_full]
        mass_vis = float(hv_raw.sum())
        visual_ratio = mass_vis / mass_all
        if visual_ratio < min_visual_ratio:
            continue

        p = torch.softmax(hv_raw / temp, dim=-1)
        if p.numel() == 1:
            focus = 1.0
        else:
            focus = 1.0 - (
                -((p * (p + 1e-12).log()).sum() / math.log(p.numel())).item()
            )
        score = (visual_ratio ** gamma) * focus

        tok_txt = (
            tokenizer.decode(full_ids[0, q_abs : q_abs + 1], skip_special_tokens=True)
            if (tokenizer is not None and full_ids is not None)
            else ""
        )
        if score > best[2]:
            best = (q_abs, tok_txt, score, visual_ratio, focus)
    return best


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
    grid_h, grid_w, grid_thw = infer_grid_shape(indices.numel(), extra_inputs)
    return VisualTokenInfo(
        mask=vis_mask,
        indices=indices,
        grid_h=grid_h,
        grid_w=grid_w,
        grid_thw=grid_thw,
    )


def infer_grid_shape(
    num_visual_tokens: int, extra_inputs: Dict[str, torch.Tensor]
) -> Tuple[int, int, Optional[List[Tuple[int, int, int]]]]:
    grid_thw: Optional[List[Tuple[int, int, int]]] = None
    if "image_grid_thw" in extra_inputs:
        thw = extra_inputs["image_grid_thw"]
        thw = thw[0].detach().cpu()
        dims: List[Tuple[int, int, int]] = []
        if thw.ndim == 1:
            values = [int(v) for v in thw.tolist()]
            if len(values) >= 3:
                dims.append((values[0], values[1], values[2]))
        else:
            for row in thw:
                vals = [int(v) for v in row.tolist()]
                if len(vals) >= 3:
                    dims.append((vals[-3], vals[-2], vals[-1]))
        dims = [(max(0, t), max(0, h), max(0, w)) for t, h, w in dims if h > 0 and w > 0]
        if dims:
            total = sum(t * h * w for t, h, w in dims)
            if total == num_visual_tokens:
                grid_thw = dims
                if len(dims) == 1:
                    return dims[0][1], dims[0][2], grid_thw
                largest = max(dims, key=lambda item: item[1] * item[2])
                return largest[1], largest[2], grid_thw
            LOGGER.warning(
                "image_grid_thw=%s totals %d tokens but located %d visual tokens; falling back.",
                dims,
                total,
                num_visual_tokens,
            )
    grid_h, grid_w = _factorized_grid(num_visual_tokens)
    LOGGER.info(
        "Inferred grid shape: %dx%d=%d for %d visual tokens",
        grid_h,
        grid_w,
        grid_h * grid_w,
        num_visual_tokens,
    )
    return grid_h, grid_w, grid_thw


def _factorized_grid(num_visual_tokens: int) -> Tuple[int, int]:
    if num_visual_tokens <= 0:
        return 1, 1
    root = int(math.sqrt(num_visual_tokens))
    for h in range(root, 0, -1):
        if num_visual_tokens % h == 0:
            return h, num_visual_tokens // h
    return 1, num_visual_tokens


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
    temp: float = 0.7,
    head_pool: str = "weighted",
    visual_ratio_gate: float = 0.12,
) -> torch.Tensor:
    num_layers = len(attentions)
    if layer_index < 0:
        layer_index += num_layers
    if not (0 <= layer_index < num_layers):
        raise IndexError(f"Layer index {layer_index} is out of range for {num_layers} layers.")
    attn_layer = attentions[layer_index]
    if attn_layer is None:
        raise RuntimeError(f"Attention at layer {layer_index} is None; cannot compute heatmap.")
    attn_layer = attn_layer[0].to(torch.float32)

    if query_index >= attn_layer.shape[1]:
        raise IndexError(
            f"Query index {query_index} is out of range for sequence length {attn_layer.shape[1]}."
        )
    if vis_info.mask.shape[0] != attn_layer.shape[2]:
        raise ValueError("Visual token mask length does not match attention key dimension.")

    pooled = pooled_over_heads_fullK(attn_layer, query_index, vis_info.mask, head_pool=head_pool)
    mass_all = pooled.sum().clamp_min(1e-6)
    hv_raw = pooled[vis_info.mask]
    visual_ratio = float(hv_raw.sum() / mass_all)
    LOGGER.info("visual_ratio(query) = %.3f (gate=%.3f)", visual_ratio, visual_ratio_gate)

    if visual_ratio >= visual_ratio_gate:
        hv = torch.softmax(hv_raw / temp, dim=-1)
    else:
        hv = hv_raw / mass_all

    expected = vis_info.grid_h * vis_info.grid_w
    if expected == hv.numel():
        return hv.view(vis_info.grid_h, vis_info.grid_w)
    if vis_info.grid_thw:
        reshaped = reshape_multiscale_heatmap(hv, vis_info.grid_thw)
        if reshaped is not None:
            return reshaped
    LOGGER.warning(
        "grid_h*grid_w (%d) != N_vis (%d); fallback to 1xN.",
        expected,
        hv.numel(),
    )
    return hv.view(1, -1)


def single_token_heatmap(
    attentions,
    layer_index,
    query_index,
    vis_info,
    head_pool="weighted",
    temp=0.7,
    vr_gate=0.12,
    return_vr=True,
):
    H = compute_attention_heatmap(
        attentions,
        layer_index,
        query_index,
        vis_info,
        temp=temp,
        head_pool=head_pool,
        visual_ratio_gate=vr_gate,
    )
    attn_layer = attentions[layer_index][0].to(torch.float32)
    fullK = pooled_over_heads_fullK(attn_layer, query_index, vis_info.mask, head_pool=head_pool)
    vr = float(fullK[vis_info.mask].sum() / fullK.sum().clamp_min(1e-6))
    return (H, vr) if return_vr else H


def fuse_soft_or(maps):
    out = maps[0].clone()
    for m in maps[1:]:
        out = 1 - (1 - out.clamp(0, 1)) * (1 - m.clamp(0, 1))
    return out


def phrase_heatmap(
    attentions,
    layer_index,
    abs_start,
    abs_end,
    vis_info,
    head_pool="weighted",
    temp=0.7,
    vr_gate=0.12,
    fuse="soft_or",
):
    maps: List[torch.Tensor] = []
    vrs: List[float] = []
    for q_abs in range(abs_start, abs_end + 1):
        H, vr = single_token_heatmap(
            attentions,
            layer_index,
            q_abs,
            vis_info,
            head_pool=head_pool,
            temp=temp,
            vr_gate=vr_gate,
            return_vr=True,
        )
        maps.append(H)
        vrs.append(vr)
    if len(maps) == 1:
        return maps[0]
    if fuse == "soft_or":
        return fuse_soft_or(maps)
    else:
        weights = torch.tensor(vrs, dtype=torch.float32)
        weights = (weights.clamp_min(1e-6) ** 2)
        weights = weights / weights.sum()
        fused = sum(weights[i] * maps[i] for i in range(len(maps)))
        return fused


def normalize_heatmap(heatmap: torch.Tensor) -> np.ndarray:
    heat = heatmap.detach().cpu().float().numpy()
    heat = np.maximum(heat, 0.0)
    hmin, hmax = float(heat.min()), float(heat.max())
    if hmax > hmin:
        return (heat - hmin) / (hmax - hmin + 1e-6)
    return heat


def reshape_multiscale_heatmap(
    hv: torch.Tensor, grid_thw: Sequence[Tuple[int, int, int]]
) -> Optional[torch.Tensor]:
    flat_total = hv.numel()
    expected = sum(t * h * w for t, h, w in grid_thw)
    if expected != flat_total:
        LOGGER.warning(
            "grid_thw totals %d tokens but received %d attention weights; cannot reshape multiscale map.",
            expected,
            flat_total,
        )
        return None

    max_h = max(h for _, h, _ in grid_thw)
    max_w = max(w for _, _, w in grid_thw)
    if max_h <= 0 or max_w <= 0:
        return None

    canvas = torch.zeros((max_h, max_w), dtype=hv.dtype, device=hv.device)
    counts = torch.zeros((max_h, max_w), dtype=hv.dtype, device=hv.device)
    offset = 0
    for t, h, w in grid_thw:
        num = t * h * w
        if num == 0:
            continue
        chunk = hv[offset : offset + num]
        offset += num
        chunk = chunk.view(t, h, w)
        # Collapse temporal dimension if present
        chunk = chunk.mean(dim=0)
        chunk = chunk.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            chunk,
            size=(max_h, max_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        canvas += resized
        counts += torch.ones_like(resized)

    mask = counts > 0
    if not mask.any():
        return None
    canvas[mask] = canvas[mask] / counts[mask]
    if not mask.all():
        fill_value = canvas[mask].mean()
        canvas = torch.where(mask, canvas, fill_value)
    return canvas


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
        if args.use_constraints and args.target_word:
            gen_out = generate_with_constraints(
                model,
                processor.tokenizer,
                inputs_on_device,
                args.target_word,
            )
        else:
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

    if getattr(args, "dump_gen_tokens", False):
        debug_dump_generated_tokens(processor.tokenizer, gen_ids)

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
    if chosen_layer == num_layers - 1:
        LOGGER.warning(
            "Using the LAST layer for visualization is prone to attention sinks; prefer mid layers."
        )
    LOGGER.info("Using layer %d/%d for visualization.", chosen_layer, num_layers)

    tokenizer = processor.tokenizer

    LOGGER.debug("Raw text: %s", raw_text[:200] + "..." if len(raw_text) > 200 else raw_text)
    LOGGER.debug("Input IDs shape: %s", input_ids.shape)
    LOGGER.debug("Number of visual tokens: %d", vis_info.mask.sum().item())

    if args.target_word:
        test_encoding = tokenizer.encode(args.target_word, add_special_tokens=False)
        LOGGER.debug("Target word '%s' encodes to: %s", args.target_word, test_encoding)

    prompt_len = int(inputs_on_device["input_ids"].shape[1])
    full_len = int(full_ids.shape[1])

    vis_mask_full = torch.zeros(full_len, dtype=torch.bool, device=full_ids.device)
    vis_mask_full[:original_seq_len] = vis_info.mask

    vis_info_for_attn = VisualTokenInfo(
        mask=vis_mask_full,
        indices=vis_mask_full.nonzero(as_tuple=True)[0],
        grid_h=vis_info.grid_h,
        grid_w=vis_info.grid_w,
        grid_thw=vis_info.grid_thw,
    )

    heatmap_tensor: torch.Tensor
    rel_end, matched = find_target_by_tokens(tokenizer, gen_ids, args.target_word)
    if rel_end is not None:
        ids = gen_ids[0].tolist()
        pieces = [
            tokenizer.decode(gen_ids[0, i : i + 1], skip_special_tokens=True)
            for i in range(len(ids))
        ]
        norm = [_norm_text(p) for p in pieces]
        target_norm = _norm_text(args.target_word)
        start_rel = rel_end
        for j in range(rel_end, -1, -1):
            buf = "".join(norm[j : rel_end + 1])
            if buf == target_norm:
                start_rel = j
                break
        abs_start = prompt_len + int(start_rel)
        abs_end = prompt_len + int(rel_end)
        LOGGER.info(
            "Matched GENERATED phrase: '%s' at abs [%d, %d].",
            matched,
            abs_start,
            abs_end,
        )
        heatmap_tensor = phrase_heatmap(
            attentions,
            chosen_layer,
            abs_start,
            abs_end,
            vis_info_for_attn,
            head_pool=args.head_pool,
            temp=args.temp,
            vr_gate=args.visual_ratio_gate,
            fuse="soft_or",
        )
        query_index = abs_end
    else:
        LOGGER.warning(
            "Target phrase not found in generated text; falling back to auto-pick strategy."
        )
        q_abs, tok_txt, score, vr, foc = pick_most_focused_generated_token(
            attentions,
            chosen_layer,
            prompt_len,
            full_len,
            vis_mask_full,
            head_pool=args.head_pool,
            tokenizer=tokenizer,
            full_ids=full_ids,
            gamma=2.0,
            min_visual_ratio=args.visual_ratio_gate,
            temp=args.temp,
        )
        if q_abs is not None:
            query_index = int(q_abs)
            LOGGER.info(
                "Auto-picked token: '%s' at abs %d (score=%.3f, vr=%.3f, focus=%.3f).",
                tok_txt,
                query_index,
                score,
                vr,
                foc,
            )
            heatmap_tensor = single_token_heatmap(
                attentions,
                chosen_layer,
                query_index,
                vis_info_for_attn,
                head_pool=args.head_pool,
                temp=args.temp,
                vr_gate=args.visual_ratio_gate,
                return_vr=False,
            )
        else:
            query_index = select_target_token(
                tokenizer,
                raw_text,
                input_ids,
                vis_info.mask,
                args.target_word,
            )
            fallback_txt = tokenizer.decode(
                full_ids[0, query_index : query_index + 1], skip_special_tokens=True
            )
            LOGGER.info("Fallback to INPUT token: '%s' at abs %d.", fallback_txt, query_index)
            heatmap_tensor = single_token_heatmap(
                attentions,
                chosen_layer,
                int(query_index),
                vis_info_for_attn,
                head_pool=args.head_pool,
                temp=args.temp,
                vr_gate=args.visual_ratio_gate,
                return_vr=False,
            )

    last_vis = int(vis_info.indices.max().item())
    LOGGER.info("Last visual token index: %d.", last_vis)
    assert query_index > last_vis, (
        f"query_index({query_index}) must be AFTER last visual token ({last_vis})."
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
