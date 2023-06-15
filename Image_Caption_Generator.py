#!/usr/bin/env python
# coding: utf-8

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# This file helps in Generating caption for the image provided by the user

#defining the name of our huggingface🤗 model
model_name = "Salesforce/blip-image-captioning-base"
image_processor_name = "Salesforce/blip-image-captioning-base"


# ### 1. Definig our HugginFace🤗 Model and initializing it

# initializing our Blip-image-caption model 
# into tokens so that we can access the caption.
def ModelInitializer(model_name, image_processor):
    
    processor = BlipProcessor.from_pretrained(image_processor)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    # checking for our available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, processor


# 2. Defining our Image processor

#this function will convert our image into RGB format and use our image processor to convert our image into suitable
# format to fed it into our transformer and convert them into pytorch tensors
# and to generate our subtle image captions
def CaptionGenerator(image_file_path, caption_model, image_processor):
    #opening our image with the help of PIL library

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_name = image_file_path
    img = Image.open(img_name)
    
    # converting our image in "RGB" format
    if img.mode != 'RGB':
        img = img.convert(mode="RGB")
    
    # using our above defined image processor to generate decoded image captions
    decoded_caption = image_processor(img, return_tensors="pt")

    # encoding our generated decoded caption
    out = caption_model.generate(**decoded_caption)
    return image_processor.decode(out[0], skip_special_tokens=True)



