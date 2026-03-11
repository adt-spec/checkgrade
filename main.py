import os
import json
import urllib.request
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

import firebase_admin
from firebase_admin import credentials, storage
from google import genai

# --- 1. INITIALIZE FIREBASE STORAGE ---
try:
    # Ensure serviceAccountKey.json is in your GitHub root
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'checkgrade-by-adt.firebasestorage.app'
    })
    bucket = storage.bucket()
    print("✅ Firebase Admin initialized securely.")
except Exception as e:
    print(f"⚠️ Firebase initialization failed (check your JSON key): {e}")

# --- 2. INITIALIZE GEMINI (VIA RENDER ENVIRONMENT VARIABLE) ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable not found!")
client = genai.Client(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*50)
print("✅✅✅ DUAL-ENGINE AI SERVER WITH CLOUD SHADOW LOGGER ✅✅✅")
print("="*50 + "\n")

# --- HEALTH CHECK ROUTE ---
@app.get("/")
def read_root():
    return {"status": "CheckGrade AI Server is Live and Connected to Firebase"}

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
        
        # 2. Open Images (Keep raw bytes for Firebase Upload later)
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

            # --- CLOUD SHADOW LOGGING LOGIC ---
            # Automatically save this interaction to Firebase to build your custom dataset
            try:
                if 'bucket' in globals():
                    session_id = str(uuid.uuid4())[:8]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    score = result_data.get("score", 0)
                    
                    # Group them by score in Firebase (e.g., /automl_training_dataset/score_4.5/...)
                    base_path = f"automl_training_dataset/score_{score}/{timestamp}_{session_id}"
                    
                    # 1. Upload the raw image
                    img_blob = bucket.blob(f"{base_path}/actual_scan.jpg")
                    img_blob.upload_from_string(image_data, content_type='image/jpeg')
                    
                    # 2. Upload the Gemini JSON label
                    json_blob = bucket.blob(f"{base_path}/ai_label.json")
                    json_blob.upload_from_string(json.dumps(result_data, indent=4), content_type='application/json')
                    
                    print(f"📊 Cloud Shadow Logged: Saved to Firebase Storage -> {base_path}")
                else:
                    print("⚠️ Firebase not initialized. Skipping cloud log.")
            except Exception as log_error:
                print(f"⚠️ Cloud Shadow Logging failed: {log_error}")

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
