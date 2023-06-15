# Defining out Imports
from flask import Flask, render_template, request
from Image_Caption_Generator import ModelInitializer, CaptionGenerator
from Catchy_Caption_Generator import TextGenerationPipeline, CreatePrompt, CatchyCaption
import torch
from PIL import Image

#initialising out flask app
app = Flask(__name__, template_folder='template')

# creating route to render our html file
@app.route('/')
def index():
    return render_template('index.html')

# function to create our catchy captions with the help of imported py files
def create_catchy_captions(image_file_path):

    # calling create caption function
    output_caption = CaptionGenerator(image_file_path,caption_model, image_processor)

    # now we'll use our catchy caption generator to connvert output_caption into catchy caption
    # creating our prompt
    input_prompt = CreatePrompt(output_caption)
    output_catchy_captions = CatchyCaption(text_generator_pipeline, input_prompt)

    return output_catchy_captions

# creating route to upload our image on the site 

@app.route('/upload', methods=['POST'])
def upload():
    if 'image-upload' in request.files:
        file = request.files['image-upload']
        # Process the uploaded file here (e.g., save it to a folder, perform further operations)
        file_url = "static/uploads/" + file.filename
        file.save(file_url)
        image_url = file_url

        final_caption = create_catchy_captions(image_url).split(" m.")[0]

        # Display image with generated caption on Flask App
        return str(image_url +"----"+final_caption)
    
    return 'Upload failed!'  

if __name__=='__main__':

    # defining name of our huggingface models
    model_name_for_image_caption= "Salesforce/blip-image-captioning-base"
    image_processor_name = "Salesforce/blip-image-captioning-base"
    model_name_for_catchy_caption = "EleutherAI/gpt-neo-1.3B"
    caption_model, image_processor= ModelInitializer( model_name_for_image_caption, image_processor_name)
    text_generator_pipeline = TextGenerationPipeline(model_name_for_catchy_caption) 
    app.run(debug = True)