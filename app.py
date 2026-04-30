from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import time
import card_builder

app = FastAPI()

app.mount("/output", StaticFiles(directory="output"), name="output")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate_cards(
    request: Request,
    bride_name: str = Form(...),
    groom_name: str = Form(...),
    date: str = Form(...),
    venue: str = Form(...)
):
    # Pass data to card builder
    card_builder.run_pipeline(bride_name, groom_name, date, venue)
    
    timestamp = int(time.time())
    variants = [f"/output/variant_{i}.jpg?t={timestamp}" for i in range(1, 6)]
    
    return templates.TemplateResponse("results.html", {
        "request": request, 
        "variants": variants,
        "bride_name": bride_name,
        "groom_name": groom_name
    })

@app.get("/editor", response_class=HTMLResponse)
async def editor_view(request: Request, image: str = ""):
    return templates.TemplateResponse("editor.html", {"request": request, "image": image})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
