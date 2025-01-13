import os
import argparse
import gc
import glob
import json
from pathlib import Path

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
from configuration_omchat import InternVisionConfig
from modeling_omchat import OmChatForConditionalGeneration
from processing_omchat import OmChatProcessor
from image_processing_omchat import  OmChatImageProcessor
from transformers import SiglipImageProcessor, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "image_tower.": "vision_tower.",
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


def convert_omchat_to_hf(pytorch_dump_folder_path, text_model_path, image_model_path, old_state_dict_id, push_to_hub=False):
    # load original config
    file_path = os.path.join(old_state_dict_id, "config.json")

    # read json
    with open(file_path) as f:
        data = json.load(f)
        print(data)
    vision_model_id = data["mm_vision_tower"]
    torch.set_default_dtype(torch.float16)
    #text_config = AutoConfig.from_pretrained(text_model_path)
    text_config = AutoConfig.from_pretrained(text_model_path)
    print (text_config) #'vocab_size': 151668
    text_config.vocab_size = 151668
    use_fast = True
    #tokenizer = AutoTokenizer.from_pretrained(text_model_path, use_fast=use_fast)
    tokenizer = AutoTokenizer.from_pretrained(old_state_dict_id, use_fast=use_fast)
    image_processor = OmChatImageProcessor.from_pretrained(image_model_path)
    processor = OmChatProcessor(tokenizer=tokenizer, image_processor=image_processor)
    if "siglip" in vision_model_id:
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=384,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=27,
            patch_size=14,
            vision_use_head=False,
        ).to_dict()
    elif "internvit" in vision_model_id:
        vision_config = InternVisionConfig()
    else:
        vision_config = None 

    config = OmChatConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config,
        image_grid_pinpoints=data["image_grid_pinpoints"]
    )
    with init_empty_weights():
        model = OmChatForConditionalGeneration(config)

    # load original state dict
    state_dict = load_original_state_dict(old_state_dict_id)
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, assign=True, strict=False)
    model.eval()

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--t",
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--v",
        help="Hub location of the vision model",
    )
    parser.add_argument(
        "--o",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--i",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    args = parser.parse_args()
    #convert_llava_llama_to_hf(args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id)
    convert_omchat_to_hf( 
            text_model_path=args.t, 
            pytorch_dump_folder_path=args.o, 
            image_model_path=args.v,
            old_state_dict_id=args.i)
