# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llama_lora_vl_flash_attn_monkey_patch import replace_llama_lora_vl_attn_with_flash_attn

replace_llama_lora_vl_attn_with_flash_attn()

from llava.train.train_lora_vl import train

if __name__ == "__main__":
    train()
