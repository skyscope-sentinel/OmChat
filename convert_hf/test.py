import argparse
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
import requests
import torch


def test_model(model_name):
    """Test the OmChat model with the specified model checkpoint."""
    # Load the model and processor
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        return_token_type_ids=True
    )

    # Load test images
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image2 = Image.open("/data2/omchat_dev/official/OmChat/images/extreme_ironing.jpg")

    # Define the prompt
    prompt = "What's the content of the image?"
    # prompt = "What are the differences between two images. image1: <image> and image2: <image>"

    # Prepare the input
    inputs = processor(
        text=prompt,
        images=image,
        system_prompt=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        ),
        return_tensors="pt"
    ).to("cuda")

    # Perform inference
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=1,
            do_sample=False,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    # Decode and print the output
    outputs = processor.tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:]).strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the OmChat model with a specified checkpoint")
    parser.add_argument(
        "--model_name",
        default="checkpoints/omchat_temp_multi_hf",
        help="Path to the model checkpoint directory (default: 'checkpoints/omchat_temp_multi_hf')"
    )

    args = parser.parse_args()
    test_model(args.model_name)

