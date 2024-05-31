from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse,  JSONResponse
import replicate
import os
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from chatgemini import ChatGoogleGenerativeAI
import google.generativeai as genai

app = FastAPI()

load_dotenv()
replicate.api_token = os.getenv("REPLICATE_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_lyrics(prompt):
    # Initialize the Gemini client with the Gemini Pro model
    # client = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    # Make a request to the Gemini API to generate lyrics
    response = chat.send_message(prompt,stream=True)
        # messages=[
        #     {
        #         "role": "system",
        #         "content": "You are a music lyrics writer and your task is to write lyrics of music under 30 words based on user's prompt. Just return the lyrics and nothing else."
        #     },
        #     {
        #         "role": "user",
        #         "content": prompt
        #     }
        # ],
        # max_tokens=50,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0
    #     prompt
    # )
    
    # Extract the generated lyrics from the response
    # output = response.choices[0].message.content
    # cleaned_output = output.replace("\n", " ")
    formatted_lyrics = f"♪ {response} ♪"
    print(formatted_lyrics)
    return formatted_lyrics

# # Example usage:
# prompt = "Tell me about the stars"
# lyrics = generate_lyrics(prompt)
# print(lyrics)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-music")
async def generate_music(prompt: str = Form(...), duration: int = Form(...)):
    lyrics = generate_lyrics(prompt)
    prompt_with_lyrics = lyrics
    print(prompt_with_lyrics)
    output = replicate.run(
        "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
        input={
            "prompt": prompt_with_lyrics,
            "text_temp": 0.7,
            "output_full": False,
            "waveform_temp": 0.7
        }
    )
    print(output)
    music_url = output['audio_out']
    music_path_or_url = music_url
    
    print(music_path_or_url)
    return JSONResponse(content={"url": music_path_or_url})
