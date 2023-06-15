#!/usr/bin/env python
# coding: utf-8

# This file will convert the generated caption from the Image_caption_generator into a catchy caption


# for this task huggingface pipline in used which creates a Text2Text generation
# pipeline that can use different huggingface🤗 transformers


# 1. Defining our function for initializing our HuggingFace🤗 Pipeline with transformer model
from transformers import pipeline


model_name = "EleutherAI/gpt-neo-1.3B"

def TextGenerationPipeline(model_name):
    generator = pipeline("text-generation", model_name)
    return generator


# 2. Now we'll create a prompt for our TextGeneration model

# here first to give our model basic understanding of the text to generate, we'll provide some examples in the prompt itself
# so that it can generate catchy phrases for the given input,
# in the end of the prompt we'll add our own caption to convert it and get catchy phrases in return
def CreatePrompt(generate_normal_caption):

    file1 = open(r"C:\Users\Pratham\imgae_caption_ui\prompt_for_catchy_phrase.txt","r")
    prompt = file1.read()+"""

    l. "{fa}" :
      1. """.format(fa = generate_normal_caption)
    
    return prompt


# 3.This function will use our model to and uses our prompt to generate a catchy caption for the user image

#this functions takes our huggingface pipeline and our prompt as input

def CatchyCaption(generator, prompt):
    output_caption = generator(prompt, max_length = 1600, do_sample=True, temperature=0.9)
    # max_length defines how long our output is going to be and the output includes the given prompt too so we have to keep a big
    # max_length
    # do_sample tells us whether to do sampling or not
    # temperature is the value used to model next set of probabilites of text to generate
    print(output_caption)
    return (output_caption[0]["generated_text"][len(prompt)-4:])
    

