import os
import argparse
import gc
import glob
import json
from pathlib import Path
import shutil

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer
)
from configuration_omchat import OmChatConfig
from transformers import SiglipVisionConfig
from modeling_omchat import OmChatForConditionalGeneration
from processing_omchat import OmChatProcessor
from transformers import AutoImageProcessor

KEYS_TO_MODIFY_MAPPING = {
    "image_tower": "vision_tower",
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
}

def load_original_state_dict(directory_path):
    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.float16)
    return new_state_dict


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def copy_files_to_output_folder(output_folder):
    """Copy all .py and .json files to the output folder, excluding 'omchat_main.py'."""
    for file in glob.glob("*.py") + glob.glob("*.json"):
        if nor file in ["convert_main.py","test.py"]:
            shutil.copy(file, os.path.join(output_folder, file))


def update_import_in_modeling_file(file_path):
    """Update the import in modeling_omchat.py to use relative import."""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    with open(file_path, "r") as file:
        content = file.read()
    
    # Replace the specific line for the import
    updated_content = content.replace(
        "from configuration_omchat import OmChatConfig",
        "from .configuration_omchat import OmChatConfig"
    )
    
    with open(file_path, "w") as file:
        file.write(updated_content)
    print(f"Updated import in {file_path}")

def convert_omchat_to_hf(filepath, output_folder, text_model_id, image_model_id):
    # Ensure the output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Copy relevant files to the output folder

    # Read config.json
    with open(os.path.join(filepath, "config.json")) as f:
        data = json.load(f)
        print(data)

    vision_model_id = data["mm_vision_tower"]
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)
    use_fast = True
    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=use_fast)
    image_processor = AutoImageProcessor.from_pretrained(image_model_id)
    processor = OmChatProcessor(tokenizer=tokenizer, image_processor=image_processor)
    vision_config = SiglipVisionConfig(
        hidden_size=1152,
        image_size=384,
        intermediate_size=4304,
        num_attention_heads=16,
        num_hidden_layers=27,
        patch_size=14,
        vision_use_head=True,
    ).to_dict()
    text_config.vocab_size = 151649
    config = OmChatConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config,
        image_grid_pinpoints=data["image_grid_pinpoints"]
    )
    with init_empty_weights():
        model = OmChatForConditionalGeneration(config)

    # Load original state dict
    state_dict = load_original_state_dict(filepath)
    state_dict = convert_state_dict_to_hf(state_dict)
    unwanted_keys = [
        "vision_tower.vision_model.head.attention.in_proj_bias",
        "vision_tower.vision_model.head.attention.in_proj_weight",
        "vision_tower.vision_model.head.attention.out_proj.bias",
        "vision_tower.vision_model.head.attention.out_proj.weight",
        "vision_tower.vision_model.head.layernorm.bias",
        "vision_tower.vision_model.head.layernorm.weight",
        "vision_tower.vision_model.head.mlp.fc1.bias",
        "vision_tower.vision_model.head.mlp.fc1.weight",
        "vision_tower.vision_model.head.mlp.fc2.bias",
        "vision_tower.vision_model.head.mlp.fc2.weight",
        "vision_tower.vision_model.head.probe",
    ]

    model.load_state_dict(state_dict, assign=True)
    model.eval()

    model.save_pretrained(output_folder)
    processor.save_pretrained(output_folder)
    # Copy relevant files to the output folder
    copy_files_to_output_folder(output_folder)
    update_import_in_modeling_file(os.path.join(output_folder,"modeling_omchat.py"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OmChat model to HF format")
    parser.add_argument("--model_name", required=True, help="Path to the original OmChat model directory")
    parser.add_argument("--output_folder", required=True, help="Path to save the converted HF model")
    parser.add_argument(
        "--text_model_id", 
        default="/data2/omchat_dev/omchat/resources/Qwen2-7B", 
        help="Text model ID (default: '/data2/omchat_dev/omchat/resources/Qwen2-7B')"
    )
    parser.add_argument(
        "--image_model_id", 
        default="/data2/omchat_dev/omchat/resources/siglip-so400m-patch14-384", 
        help="Image model ID (default: '/data2/omchat_dev/omchat/resources/siglip-so400m-patch14-384')"
    )

    args = parser.parse_args()
    convert_omchat_to_hf(args.model_name, args.output_folder, args.text_model_id, args.image_model_id)

