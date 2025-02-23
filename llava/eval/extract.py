
from transformers import AutoConfig, LlamaConfig

class LlavaConfig(LlamaConfig):
    model_type = "llava"

AutoConfig.register("llava", LlavaConfig)

import pickle
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.model.llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

import numpy as np
from PIL import Image
import random
import math

from torchvision.datasets import ImageFolder


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)

# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

def target_transform(target):
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/, class 0 is background
    category_id_list = np.zeros(90)
    for item in target:
        category_id_list[item['category_id']-1] = 1
    return category_id_list

def collate(batch):
    '''
    :batch:
    :return:
    images : (tensor)
    targets : (list) [(tensor), (tensor)]
    '''
    targets = []
    images = []
    for x in batch:
        images.append(x[0])
        targets.append(x[1])
    #print(targets)
    return images, torch.Tensor(np.asarray(targets))

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.model.vision_tower[0]
        vision_tower.to(device='cuda', dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower, torch_dtype=torch.float16)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    my_dataset = ImageFolder(root=args.image_folder)

    data_loader = torch.utils.data.DataLoader(
            dataset=my_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=collate)

    llm_features_list = []
    target_list = []
    num_layer = 1
    base_prompt = 'Describe the items in the image:'

    for i, (images, targets) in enumerate(tqdm(data_loader)):

        qs = base_prompt
        cur_prompt = qs

        batch_size = len(images)
        image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
        images = image_tensor.half().cuda()
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
        else:
            qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        cur_prompt = cur_prompt + '\n' + '<image>'

        if args.conv_mode == 'simple_legacy':
            qs += '\n\n### Response:'
        #assert gt_ans['from'] == 'gpt'
        # conv = default_conversation.copy()
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        #print(prompt)
        # A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: Hi!###Assistant: Hi there! How can I help you today?###Human: Describe the items in the image:<im_start><im_patch><im_patch><im_end>###
        prompt_list = []
        for idx in range(len(targets)):
            prompt_list.append(prompt+'Assistant: ')
            #prompt_list.append(prompt+'Assistant: '+targets[idx])

        inputs = tokenizer(prompt_list)

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=False,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])


        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        #output_prompt = tokenizer.batch_decode(output_ids)[0]
        #token_list.append(tokenizer.tokenize(output_prompt))

        llm_features = []
        with torch.inference_mode():
            
            outputs = model.model(output_ids, images=images, output_hidden_states=True)

            llm_features = outputs.hidden_states[-1].mean(1).detach().cpu()
            
            llm_features_list.append(llm_features)
            target_list.append(targets.detach().cpu())

    llm_features_list = torch.cat(llm_features_list).cpu().numpy()
    target_list = torch.cat(target_list).cpu().numpy()

    np.save(os.path.join(args.save_dir, "llm_features.npy"), llm_features_list)
    np.save(os.path.join(args.save_dir, "target.npy"), target_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--exp_name", type=str, default="")
    args = parser.parse_args()

    eval_model(args)