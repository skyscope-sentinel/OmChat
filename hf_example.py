from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
import requests
import torch
from transformers import TextStreamer

model = AutoModel.from_pretrained("omchat_fk83_hf",trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
processor = AutoProcessor.from_pretrained("omchat_fk83_hf",trust_remote_code=True)

#image = Image.open("/ssd/ljj/proj/omchateval_normal/LMUData/images/MathVista_MINI/190.jpg")
#image = Image.open("/ssd/ljj/proj/omchateval/LMUData/images/MathVista_MINI/498.jpg")
image = Image.open("images/extreme_ironing.jpg")
#prompt = 'Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Move the ruler to measure the length of the nail to the nearest inch. The nail is about (_) inches long.'
#prompt = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: Is the water half full?\nChoices:(A) Yes\n(B) No"
prompt ="describe image"
inputs = processor(text=prompt, system_prompt="hello", images=image, return_tensors="pt").to("cuda")
with torch.inference_mode():

    output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, eos_token_id=model.generation_config.eos_token_id,  pad_token_id=processor.tokenizer.pad_token_id)

outputs = processor.tokenizer.decode(output_ids[0, inputs.input_ids.shape[1] :]).strip()

"""
outputs1 = processor.tokenizer.decode([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,         151645,    198, 151644,    872,    198,   100,    198,   3400,     25,           100,    198,   3400,     25,   100,    198,   3838,    374,    856,            304,    279,  23606,     30, 151645,    198, 151644,  77091,    198]).strip()
outputs2 = processor.tokenizer.decode([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198,    198, 151644,    872,    271,   100,    198,   3400,
             25,   100,    198,   3400,     25,   100,    198,   3838,    374,
            279,   4226,    311,    279,   2086,  23606,    389,    279,   1290,
             30, 151645,    198, 151644,  77091,    198]).strip()
print (outputs1)
print ("*")
print (outputs2)
"""
print (outputs)
#print(processor.tokenizer.decode([7110, 77]))
#print(processor.tokenizer.encode("\n"))
#print(processor.tokenizer.decode([624]))
