import os
import uuid
import json
import io
import threading
import requests
from datetime import datetime
from PIL import Image

import firebase_admin
from firebase_admin import credentials, storage
from google import genai
from firebase_functions import https_fn

# Firebase is deferred entirely to prevent deployment hang

# Cache to prevent downloading standard images on every request
STANDARD_IMG_CACHE = {}

def resize_image(image: Image.Image, max_dim: int = 768) -> Image.Image:
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.Resampling.BILINEAR)
    return image

def log_to_firebase(image_data, result_data):
    try:
        # Initialize Firebase lazily
        if not firebase_admin._apps:
            firebase_admin.initialize_app(options={
                'storageBucket': 'checkgrade-by-adt.firebasestorage.app'
            })
        bucket = storage.bucket()
        
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        score = result_data.get("score", 0)
        base_path = f"automl_training_dataset/score_{score}/{timestamp}_{session_id}"
        
        img_blob = bucket.blob(f"{base_path}/actual_scan.jpg")
        img_blob.upload_from_string(image_data, content_type='image/jpeg')
        
        json_blob = bucket.blob(f"{base_path}/ai_label.json")
        json_blob.upload_from_string(json.dumps(result_data, indent=4), content_type='application/json')
        
        print(f"☁️ Cloud Shadow Logged: Saved to Firebase Storage -> {base_path}")
    except Exception as log_error:
        print(f"⚠️ Cloud Shadow Logging failed: {log_error}")

@https_fn.on_request(timeout_sec=120, memory=512, secrets=["GEMINI_API_KEY"])
def api(req: https_fn.Request) -> https_fn.Response:
    # Set CORS headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    
    # Pre-flight request handling
    if req.method == "OPTIONS":
        return https_fn.Response("", status=204, headers=headers)

    # Health check route
    if req.path == "/" or req.path == "":
        api_key_exists = "Yes" if os.environ.get("GEMINI_API_KEY") else "No"
        return https_fn.Response(json.dumps({
            "status": "CheckGrade AI Server is Live",
            "firebase_connected": "True",
            "gemini_api_key_configured": api_key_exists,
            "engine_version": "2.2-Native"
        }), status=200, headers={"Content-Type": "application/json", **headers})

    # Main API route
    if req.path == "/audit-zone":
        if req.method != "POST":
            return https_fn.Response(json.dumps({"error": "Method not allowed"}), status=405, headers={"Content-Type": "application/json", **headers})
            
        try:
            engine = req.form.get("engine", "gemini")
            standard_image_url = req.form.get("standard_image_url")
            actual_image_file = req.files.get("actual_image")

            if not standard_image_url or not actual_image_file:
                return https_fn.Response(json.dumps({"error": "Missing image or URL"}), status=400, headers={"Content-Type": "application/json", **headers})

            # Process Actual Image
            image_data = actual_image_file.read()
            actual_img_raw = Image.open(io.BytesIO(image_data))
            actual_img = resize_image(actual_img_raw)

            # Process Standard Image (Synchronous)
            url_key = standard_image_url.replace(" ", "%20")
            if url_key in STANDARD_IMG_CACHE:
                standard_img_data = STANDARD_IMG_CACHE[url_key]
            else:
                resp = requests.get(url_key, timeout=15.0)
                resp.raise_for_status()
                standard_img_data = resp.content
                STANDARD_IMG_CACHE[url_key] = standard_img_data
            
            standard_img = resize_image(Image.open(io.BytesIO(standard_img_data)))

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
                
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    return https_fn.Response(json.dumps({"error": "GEMINI_API_KEY missing"}), status=500, headers={"Content-Type": "application/json", **headers})
                
                client = genai.Client(api_key=api_key)

                models_to_try = [
                    'gemini-2.5-flash',
                    'gemini-2.0-flash', 
                    'gemini-2.0-flash-exp',
                    'gemini-1.5-flash-latest', 
                    'gemini-1.5-flash',
                    'gemini-1.5-pro'
                ]
                
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
                    return https_fn.Response(json.dumps({"error": f"AI Engine Offline. {last_error}"}), status=503, headers={"Content-Type": "application/json", **headers})
                
                result_text = response.text.replace("```json", "").replace("```", "").strip()
                result_data = json.loads(result_text)
                result_data["engine_used"] = active_model

                # Background Task using threading
                threading.Thread(target=log_to_firebase, args=(image_data, result_data)).start()

                return https_fn.Response(json.dumps(result_data), status=200, headers={"Content-Type": "application/json", **headers})

            elif engine == "automl":
                return https_fn.Response(json.dumps({
                    "score": 4.0, 
                    "feedback": "Your Custom AutoML model is not linked yet!", 
                    "analysis_type": "AutoML Preview"
                }), status=200, headers={"Content-Type": "application/json", **headers})

            else:
                return https_fn.Response(json.dumps({"error": "Invalid engine"}), status=400, headers={"Content-Type": "application/json", **headers})

        except Exception as e:
            print(f"❌ SERVER CRASH: {str(e)}")
            return https_fn.Response(json.dumps({"error": str(e)}), status=500, headers={"Content-Type": "application/json", **headers})

    # Catch-all
    return https_fn.Response("Endpoint not found", status=404, headers=headers)