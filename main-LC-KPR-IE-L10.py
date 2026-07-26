import os
import json
import uuid
import httpx
import io
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, storage
from google import genai
import openai
import base64

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
client = genai.Client(api_key=api_key) if api_key else None

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️ WARNING: OPENAI_API_KEY environment variable not found!")
openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*50)
print("✅✅✅ OPTIMIZED DUAL-ENGINE AI SERVER ✅✅✅")
print("="*50 + "\n")

def resize_image(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Resizes an image maintaining aspect ratio if any dimension exceeds max_dim."""
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def log_to_firebase(image_data, result_data):
    """Background task to log interaction to Firebase."""
    try:
        if 'bucket' in globals():
            session_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            score = result_data.get("score", 0)
            
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

# --- HEALTH CHECK ROUTE ---
@app.get("/")
def read_root():
    api_key_exists = "Yes" if os.environ.get("GEMINI_API_KEY") else "No"
    return {
        "status": "CheckGrade AI Server is Live",
        "firebase_connected": 'bucket' in globals(),
        "gemini_api_key_configured": api_key_exists,
        "engine_version": "2.1-Robust"
    }

@app.post("/api/audit-zone")
async def audit_zone(
    background_tasks: BackgroundTasks,
    actual_image: UploadFile = File(...), 
    standard_image_url: str = Form(...),
    engine: str = Form("gemini")
):
    try:
        print(f"--> Incoming Image: {actual_image.filename}")
        print(f"--> Active Engine: {engine.upper()}")
        
        # 1. Process Standard Image URL (Async)
        image_data = await actual_image.read()
        actual_img_raw = Image.open(io.BytesIO(image_data))
        actual_img = resize_image(actual_img_raw)

        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(standard_image_url.replace(" ", "%20"), timeout=15.0)
            resp.raise_for_status()
            standard_img_data = resp.content
        
        standard_img = resize_image(Image.open(io.BytesIO(standard_img_data)))

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
            
            # --- ROBUST FALLBACK ENGINE ---
            models_to_try = ['gemini-2.0-flash', 'gemini-1.5-flash']
            response = None
            last_error = ""
            active_model = ""

            for model_name in models_to_try:
                try:
                    print(f"🤖 Attempting analysis with: {model_name}")
                    response = client.models.generate_content(
                        model=model_name, 
                        contents=[prompt, standard_img, actual_img]
                    )
                    active_model = model_name
                    print(f"✅ Analysis Successful using {model_name}")
                    break
                except Exception as e:
                    last_error = str(e)
                    print(f"⚠️ Model {model_name} failed: {last_error}")
            
            if not response:
                print("❌ ALL AI MODELS FAILED")
                raise HTTPException(status_code=503, detail=f"AI Engine Offline. Please try again later. (Error: {last_error})")
            
            result_text = response.text.replace("```json", "").replace("```", "").strip()
            result_data = json.loads(result_text)
            
            # Add metadata about which model was used
            result_data["engine_used"] = active_model

            # --- OPTIMIZED: Move logging to background task ---
            background_tasks.add_task(log_to_firebase, image_data, result_data)

            return result_data

        # ==========================================
        # PATH B: VERTEX AI AUTOML ENGINE (Future)
        # ==========================================
        elif engine == "automl":
            return {
                "score": 4.0, 
                "feedback": "Your Custom AutoML model is not linked yet!", 
                "analysis_type": "AutoML Preview"
            }

        # ==========================================
        # PATH C: OPENAI ENGINE
        # ==========================================
        elif engine == "openai":
            if not openai_client:
                raise HTTPException(status_code=503, detail="OpenAI Engine Offline (No API Key).")
            
            prompt = """
            You are a strict, expert 5S Factory Auditor. 
            Compare Image 2 (Actual) against Image 1 (Standard).
            Check if they are the same room. If not, score very low (1.0).
            Score the Actual image from 0 to 5 (decimals allowed).
            Provide a short "analysis_type" (e.g., "Compliant", "Severe Clutter").
            
            Return ONLY a valid JSON object like this:
            { "score": 2.5, "feedback": "Explanation...", "analysis_type": "Needs Improvement" }
            """

            # encode images to base64
            actual_b64 = base64.b64encode(image_data).decode("utf-8")
            standard_b64 = base64.b64encode(standard_img_data).decode("utf-8")
            
            print(f"🤖 Attempting analysis with: gpt-4o")
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{standard_b64}"}
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{actual_b64}"}
                                }
                            ]
                        }
                    ],
                    response_format={ "type": "json_object" }
                )
                print("✅ Analysis Successful using gpt-4o")
                result_text = response.choices[0].message.content.strip()
                result_data = json.loads(result_text)
                result_data["engine_used"] = "gpt-4o"
                
                background_tasks.add_task(log_to_firebase, image_data, result_data)
                return result_data
            except Exception as e:
                print(f"❌ OpenAI FAILED: {str(e)}")
                raise HTTPException(status_code=503, detail=f"OpenAI Engine Offline. Please try again later. (Error: {str(e)})")

        else:
            raise ValueError("Invalid AI Engine selected.")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"\n❌ SERVER CRASH: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)