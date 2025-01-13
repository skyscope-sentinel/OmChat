import torch
import torch.nn as nn

from transformers import SiglipImageProcessor, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        #self.select_layer = args.mm_vision_select_layer
        self.select_layer = getattr(args, 'mm_vision_select_layer', -2)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name, cache_dir=self.cache_dir)

    def load_model(self,image_size=336,is_train=True):
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name, cache_dir=self.cache_dir)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, cache_dir=self.cache_dir)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs, layers=[12,16,22,23]):
        image_feature_list = []
        for l in layers:
            image_feature_list.append(image_forward_outs.hidden_states[l])
        image_features_multi = torch.cat(image_feature_list, dim=2)

        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            image_features_multi = image_features_multi[:, 1:]

        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features, image_features_multi    
    
    def forward(self, images):

        if type(images) is list:
            image_features = []
            image_features_multi = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature, image_feature_multi = self.feature_select(image_forward_out)

                image_features.append(image_feature.to(image.dtype))
                image_features_multi.append(image_feature_multi.to(image.dtype))
            return image_features_multi
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features, image_features_multi = self.feature_select(image_forward_outs)

            return image_features_multi.to(images.dtype)


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size*4

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

if __name__ == '__main__':
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained('resources/siglip-so400m-patch14-384')
    model = SiglipVisionTower('resources/siglip-so400m-patch14-384', cfg)
    print(model.hidden_size)
    # forward an image
    from PIL import Image
    image_file = 'unittest/test.jpg'
    image = Image.open(image_file).convert('RGB')
    image_tensor = model.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    print(image_tensor.shape)
    tokens = model(image_tensor)  # torch.Size([1, 3, 224, 224])
    print(tokens.shape)  # torch.Size([1, 196, 768])
    tokens = model([image_tensor[0]])  # [torch.Size([3, 224, 224])]
    print(tokens) 
