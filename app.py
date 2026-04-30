from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import time
import card_builder
from pydantic import BaseModel
import random
import base64
from io import BytesIO
from PIL import Image
try:
    from rembg import remove
except ImportError:
    remove = None

app = FastAPI()

app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")
app.mount("/Fonts", StaticFiles(directory="Fonts"), name="Fonts")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse(request=request, name="form.html", context={"request": request})

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
    
    return templates.TemplateResponse(request=request, name="results.html", context={
        "request": request, 
        "variants": variants,
        "bride_name": bride_name,
        "groom_name": groom_name
    })

@app.get("/editor", response_class=HTMLResponse)
async def editor_view(request: Request, image: str = ""):
    return templates.TemplateResponse(request=request, name="editor.html", context={"request": request, "image": image, "data": request.query_params.get("data", "")})

class RewriteRequest(BaseModel):
    text: str

@app.post("/api/magic-rewrite")
async def magic_rewrite(req: RewriteRequest):
    text = req.text.strip().lower()
    
    # Simple mock AI logic for demonstration
    rewrites = []
    if "invitation" in text or "save the date" in text:
        rewrites = [
            "Together with their families",
            "Join us for our special day",
            "You are joyfully invited",
            "Request the honor of your presence"
        ]
    elif "&" in text or "and" in text:
        rewrites = [
            req.text.replace("&", "and").replace("And", "and").upper(),
            req.text.title()
        ]
    else:
        # Generic text enhancement
        rewrites = [
            f"✨ {req.text} ✨",
            req.text.upper(),
            f"The Joyful {req.text}"
        ]
        
    # In a real app, call OpenAI/Gemini API here
    time.sleep(0.5) # Simulate AI processing time
    return {"rewritten_text": random.choice(rewrites)}

class RemoveBgRequest(BaseModel):
    image_base64: str

@app.post("/api/remove-bg")
async def remove_background(req: RemoveBgRequest):
    if remove is None:
        return {"error": "rembg is not installed on the server"}
        
    try:
        # Strip header if present
        header, encoded = req.image_base64.split(",", 1) if "," in req.image_base64 else ("", req.image_base64)
        
        # Decode image
        img_data = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_data))
        
        # Remove background
        result_img = remove(img)
        
        # Encode back to base64
        buffered = BytesIO()
        result_img.save(buffered, format="PNG")
        result_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image_base64": f"data:image/png;base64,{result_b64}"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
