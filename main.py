import os
import json
import urllib.request
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from google import genai
# Note: You will later need to run `pip install google-cloud-aiplatform` for AutoML
# from google.cloud import aiplatform 

# Paste your PERSONAL @gmail.com API key here!
client = genai.Client(api_key="AIzaSyBe7lB08tLCGuVcV6PU8Yp4rqor7LBXmXQ")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW: SHADOW LOGGER CONFIGURATION ---
# This creates a folder to store data for your future custom AutoML model
TRAINING_DIR = "automl_training_dataset"
if not os.path.exists(TRAINING_DIR):
    os.makedirs(TRAINING_DIR)
    print(f"📁 Initializing training directory: {TRAINING_DIR}")

print("\n" + "="*50)
print("✅✅✅ DUAL-ENGINE AI SERVER WITH SHADOW LOGGER ✅✅✅")
print("="*50 + "\n")

@app.post("/api/audit-zone")
async def audit_zone(
    actual_image: UploadFile = File(...), 
    standard_image_url: str = Form(...),
    engine: str = Form("gemini")
):
    try:
        print(f"--> Incoming Image: {actual_image.filename}")
        print(f"--> Active Engine: {engine.upper()}")
        
        # 1. Process Standard Image URL
        if standard_image_url.startswith('/'):
            standard_image_url = f"http://localhost:8081{standard_image_url}"
        standard_image_url = standard_image_url.replace(" ", "%20")
        
        # 2. Open Images
        image_data = await actual_image.read()
        actual_img = Image.open(io.BytesIO(image_data))

        req = urllib.request.Request(standard_image_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            standard_img_data = response.read()
        standard_img = Image.open(io.BytesIO(standard_img_data))

        # ==========================================
        # PATH A: GEMINI ENGINE (General Intelligence)
        # ==========================================
        if engine == "gemini":
            prompt = """
            You are a strict, expert 5S Factory Auditor. 
            Compare Image 2 (Actual) against Image 1 (Standard).
            Check if they are the same room. If not, score very low (1.0).
            Score the Actual image from 0 to 5 (decimals allowed).
            Provide a short "analysis_type" (e.g., "Compliant", "Severe Clutter").
            
            Return ONLY a valid JSON object like this:
            { "score": 2.5, "feedback": "Explanation...", "analysis_type": "Needs Improvement" }
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=[prompt, standard_img, actual_img]
            )
            
            result_text = response.text.replace("```json", "").replace("```", "").strip()
            result_data = json.loads(result_text)

            # --- SHADOW LOGGING LOGIC ---
            # Automatically save this interaction to build your custom dataset
            try:
                session_id = str(uuid.uuid4())[:8]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = f"{timestamp}_{session_id}"
                save_path = os.path.join(TRAINING_DIR, folder_name)
                os.makedirs(save_path)

                # Save the audit photo taken by the auditor
                actual_img.save(os.path.join(save_path, "actual_scan.jpg"))
                
                # Save Gemini's score as the "Teacher Label" for the future AI
                with open(os.path.join(save_path, "ai_label.json"), "w") as f:
                    json.dump(result_data, f, indent=4)
                    
                print(f"📊 Shadow Logged: Training data saved to {save_path}")
            except Exception as log_error:
                print(f"⚠️ Shadow Logging failed: {log_error}")

            return result_data

        # ==========================================
        # PATH B: VERTEX AI AUTOML ENGINE (Future)
        # ==========================================
        elif engine == "automl":
            print("--> Routing to Google Vertex AI AutoML Endpoint...")
            
            # Temporary mock response until you train the model with your shadow-logged data:
            return {
                "score": 4.0, 
                "feedback": "Your Custom AutoML model is not linked yet! This is a placeholder using shadow-logged data logic.", 
                "analysis_type": "AutoML Preview"
            }

        else:
            raise ValueError("Invalid AI Engine selected.")

    except Exception as e:
        print(f"\n❌ SERVER CRASH: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)