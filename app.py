import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path
import gradio as gr

load_dotenv()

API_KEY=os.getenv("API_KEY")

genai.configure(api_key=API_KEY)

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

generation_config = {
    "temperature":0.8,
    "top_p":0.9,
    "top_k":50,
    "max_output_tokens":5000,
}

model = genai.GenerativeModel(model_name="gemini-1.5-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def input_image_setup(file_location):
    if not (img := Path(file_location)).exists():
        raise FileNotFoundError(f"Image was not found: {img}")

    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": Path(file_location).read_bytes()
            }
        ]
    return image_parts


def generate_response(default_prompt, text_input, image_loc=None):
    prompt_parts = [default_prompt + text_input]
    if image_loc:
        image_prompt = input_image_setup(image_loc)
        prompt_parts.append(image_prompt[0])
    response = model.generate_content(prompt_parts)
    return response.text

default_prompt = """You are an advanced AI model specializing in suggesting cars for users based on their requirements for various factors. 
Given the requirements of the user for their next car, your task is to give them 3-4 suggestions for cars based on their requirements that they choose and the 
order of the priority of their requirements. Here are a few ways you can categorize the user's requirements. If only an image is provided, then analyze the given image and give a suggestion to the closest vehicle that you think is depicted or seen in the image.
1. Price: The cost of the car  in the chosen currency of the user. If the car is available only in a certain market without the selected currency, then convert the cost from the original currency to the selected currency
2. Economy: The fuel economy of the car in kilometers/liter. If available in different units, then convert to kilometers/liter
3. Safety: The Global NCAP star rating for the car's safety. If Global NCAP rating not available then closest regional ratings are to be provided.
4. Engine Size, type and placement: The number of cylinders, their configuration and position of the engine is to be given. Other engine options for the same car are to be provided as well.
5. Practicality: How many people the car can fit comfortably. How much bootspace the car has. How many creature comforts does the car have. How versatile are the seating options.
6. Technology: If the car has any standout tech features. If the car has ADAS. 
7. Body Type: The user will get to choose the body type of the car that they want to buy. If it's an SUV, hatchback, Sedan, Pickup, Station-wagon, or any else.
8. Fuel Type: The preferred fuel type, whether it's petrol or diesel or a hybrid or electric or hydrogen fuel cell.
9. Colour options: The preferred color for the user.
10. Advantages or plus points of the car
11. Disadvantages or known drawbacks of the car
Output should be one after another with the car name in Caps at the top. Its important points should be specified below it with the ">" symbol at each one's start. Avoid using "*" symbols
The prompted message is: """

def upload_file(files, text_input, price_input, economy_input, body_type_input, practicality_input, tech_input, fuel_type, color_input):
    file_paths = [file.name for file in files] if files else [None]
    response = generate_response(default_prompt + f"\nBodytype: {body_type_input}\n"+ f"Price: {price_input}\n"+ f"\nEconomy: {economy_input}\n"+ f"\nPracticality: {practicality_input}\n"+ f"\nTechnology: {tech_input}\n"+ f"\nFuel Type: {fuel_type}\n"+ f"\nColour Options: {color_input}\n", text_input, file_paths[0])
    return file_paths[0], response

with gr.Blocks() as demo:
    header = gr.Label("🚗DriveFinder: The perfect tool for your next car!🚗")
    text_input = gr.Textbox(placeholder="Ex:Reliable farm vehicle, Comfortable city car.",label="\bEnter your main use case and what you are mainly looking for in your next car.\b")
    price_input = gr.Textbox(placeholder="Enter in any currency required.",label="Enter the price range in which you want to select your next car.")
    economy_input = gr.Textbox(placeholder="Preferable use km/l or mpg",label="Enter the minimum fuel economy you wish your next car to have.")
    body_type_input = gr.Textbox(placeholder="Ex:Hatchback, Sedan, Coupe, SUV, Pick-Up, Van, Convertible, Station Wagon",label="Select body-type that you would like.")
    practicality_input = gr.Textbox(placeholder="Number of seats and bootspace required",label="Enter how much of a priority is bootspace to you and how many people you would like your car to seat.")
    tech_input = gr.Textbox(placeholder="Ex: Panoramic sunroof, ADAS, Air Purifier, AWD",label="Enter the technology you would like to see in your car.")
    fuel_type=gr.Textbox(placeholder="Ex:Petrol, Diesel, Hybrid, Electric, Hydrogen Fuel Cell",label="Enter your preferred fuel type.")
    color_input = gr.Textbox(placeholder="Specify colours",label="Enter your preferred first colour and second colour.")
    header = gr.Label("If you would like to upload a picture of car that needs to be identified, please do so!")
    image_output = gr.Image()
    upload_button = gr.UploadButton("Click to upload an image", file_types=["image"], file_count="multiple")
    generate_button = gr.Button("Find Cars")
    
    file_output = gr.Textbox(label="The cars that suits all/most of your needs are")
    
    def process_generate(files, text_input, price_input, economy_input, body_type_input, practicality_input, tech_input, fuel_type, color_input):
        return upload_file (files, text_input, price_input, economy_input, body_type_input, practicality_input, tech_input, fuel_type, color_input)
    
    upload_button.upload(fn=lambda files: files[0].name if files else None, inputs=[upload_button], outputs=image_output)
    generate_button.click(fn=process_generate, inputs=[upload_button, text_input, price_input, economy_input, body_type_input, practicality_input, tech_input, fuel_type, color_input], outputs=[image_output,file_output])

demo.launch(debug=True)
