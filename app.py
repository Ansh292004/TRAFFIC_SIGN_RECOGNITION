# #!/usr/bin/env python3
# """
# traffic_sign_project.py

# Single-file VS Code-ready script that:
#  - Loads gtsrb_best_model.keras
#  - Offers a menu:
#     1) Predict single image (local file)
#     2) Real-time webcam detection
#     3) Exit

# Notes:
#  - This preserves your image-prediction preprocessing & display logic (same resize/color/normalization).
#  - Real-time uses the same preprocessing so predictions align with the model.
# """

# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

# # -------------------------
# # Configuration / Settings
# # -------------------------
# MODEL_FILENAME = "gtsrb_best_model.keras"  # place this in the same folder as this script
# IMG_SIZE = (48, 48)  # kept the same as your working image upload code
# CONFIDENCE_THRESHOLD = 0.5  # used for realtime overlay (you can change this)
# # -------------------------

# # Class names (same as your earlier list)
# class_names = [
#     'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
#     'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
#     'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 tons', 'Right-of-way at the next intersection',
#     'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 tons prohibited', 'No entry',
#     'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve',
#     'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals',
#     'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
#     'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
#     'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
#     'End of no passing', 'End of no passing by vehicles over 3.5 tons'
# ]
# num_classes = len(class_names)

# # -------------------------
# # Utility / Preprocessing
# # -------------------------
# def preprocess_for_model_bgr(img_bgr, img_size=IMG_SIZE):
#     """
#     Take an image in BGR (OpenCV default), convert to RGB, resize, normalize (0-1).
#     This follows the same pipeline as your working upload-predict function.
#     Returns an array shaped (1, H, W, 3) ready for model.predict.
#     """
#     img = cv2.resize(img_bgr, img_size)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_norm = img_rgb.astype("float32") / 255.0
#     img_batch = np.expand_dims(img_norm, axis=0)
#     return img_batch

# # -------------------------
# # Load model
# # -------------------------
# def load_model(path=MODEL_FILENAME):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Model file not found: {path}\nPlease put '{path}' in the same folder as this script.")
#     # Use tf.keras.load_model to support .keras format
#     model = tf.keras.models.load_model(path)
#     print(f"✅ Loaded model from: {path}")
#     return model

# # -------------------------
# # Image prediction (kept logic consistent with your working code)
# # -------------------------
# def predict_single_image_local(model):
#     """
#     Asks user to provide (drag-drop or paste) image path in CMD,
#     then predicts using the same model.
#     """
#     print("\n📸 Please upload your traffic sign image!")
#     print("👉 You can drag & drop the image file here and press Enter:")

#     image_path = input("Image file path: ").strip().strip('"')

#     if not image_path:
#         print("⚠️ No file provided.")
#         return

#     import os
#     if not os.path.exists(image_path):
#         print("❌ File not found. Please check the path and try again.")
#         return

#     img_bgr = cv2.imread(image_path)
#     if img_bgr is None:
#         print("❌ Could not read the image.")
#         return

#     img_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_batch = preprocess_for_model_bgr(img_bgr, IMG_SIZE)

#     preds = model.predict(img_batch)
#     pred_class = int(np.argmax(preds[0]))
#     confidence = float(np.max(preds[0])) * 100.0
#     pred_name = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"

#     print(f"\n✅ Prediction: {pred_name}")
#     print(f"💪 Confidence: {confidence:.2f}%\n")

#     plt.figure(figsize=(6,6))
#     plt.imshow(img_display)
#     plt.title(f"Prediction: {pred_name}\nConfidence: {confidence:.2f}%", fontsize=14)
#     plt.axis("off")
#     plt.show()


# # -------------------------
# # Realtime webcam detection
# # -------------------------
# def realtime_detection(model):
#     """
#     Real-time webcam detection using the same preprocessing pipeline as the image predictor.
#     Displays predicted label + confidence on the video feed.
#     Press 'q' to quit.
#     """
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam. Check camera permissions and device index.")
#         return

#     # Set a comfortable frame size (can be changed)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     font = cv2.FONT_HERSHEY_SIMPLEX

#     print("Starting webcam. Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read frame from webcam. Exiting.")
#             break

#         # Preprocess the frame as the model expects
#         img_batch = preprocess_for_model_bgr(frame, IMG_SIZE)  # shape (1,H,W,3)

#         # Predict
#         preds = model.predict(img_batch)
#         pred_class = int(np.argmax(preds[0]))
#         confidence = float(np.max(preds[0]))

#         label_text = f"{pred_class}: {class_names[pred_class] if pred_class < len(class_names) else 'Unknown'}"
#         conf_text = f"{confidence*100:.2f}%"

#         # Overlay label and confidence. Use green if above threshold else red.
#         color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)

#         # Put text on top-left
#         cv2.putText(frame, label_text, (10,30), font, 0.7, color, 2, cv2.LINE_AA)
#         cv2.putText(frame, f"Conf: {conf_text}", (10,60), font, 0.7, color, 2, cv2.LINE_AA)

#         # Optionally draw a small rectangle to indicate ROI; left as minimal so predictions are frame-wide
#         # Display
#         cv2.imshow("Traffic Sign Recognition (press 'q' to quit)", frame)

#         # Exit on 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # -------------------------
# # Simple menu
# # -------------------------
# def main():
#     # Load model once
#     try:
#         model = load_model(MODEL_FILENAME)
#     except Exception as e:
#         print(str(e))
#         return
  
#     while True:
#         print("\n========= TRAFFIC SIGN RECOGNITION =========")
#         print("1. Predict from an image (local file)")
#         print("2. Real-time webcam detection")
#         print("3. Exit")
#         choice = input("Choose an option (1/2/3): ").strip()

#         if choice == "1":
#             predict_single_image_local(model)
#         elif choice == "2":
#             realtime_detection(model)
#         elif choice == "3":
#             print("Bye 👋")
#             break
#         else:
#             print("Invalid choice. Please type 1, 2, or 3.")

# if __name__ == "__main__":
#     main()
#-------------------------------------------------
import os
import sys
import uuid
import time
import json
import asyncio
import threading
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

# WebRTC
from aiortc import RTCPeerConnection, RTCSessionDescription

# -------------------- Windows asyncio policy (important) --------------------
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# -------------------- Global asyncio loop (lives for process lifetime) --------------------
GLOBAL_LOOP = asyncio.new_event_loop()

def _loop_runner(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

_thread = threading.Thread(target=_loop_runner, args=(GLOBAL_LOOP,), daemon=True)
_thread.start()

# -------------------- Config --------------------
MODEL_PATH = "gtsrb_best_model.keras"
IMG_SIZE = (48, 48)         # must match training
CONFIDENCE_THRESHOLD = 0.50 # used for UI coloring only
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}

UPLOAD_DIR = os.path.join("static", "uploads")
METRICS_IMG = os.path.join("static", "metrics", "history.png")

CLASS_NAMES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Vehicles over 3.5 tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
    'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
    'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
    'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 tons'
]

# -------------------- App --------------------
app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------- Load model --------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# -------------------- Helpers --------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_bgr(img_bgr, target_size=IMG_SIZE):
    """Resize -> BGR2RGB -> normalize -> add batch dim (1,H,W,3)."""
    img = cv2.resize(img_bgr, target_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype("float32") / 255.0
    return np.expand_dims(img_norm, axis=0)

def get_model_facts():
    if model is None:
        return {"loaded": False}
    try:
        trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        non_trainable = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        total = int(trainable + non_trainable)
        return {
            "loaded": True,
            "input_shape": str(model.input_shape),
            "num_classes": len(CLASS_NAMES),
            "total_params": total
        }
    except Exception:
        return {"loaded": True, "input_shape": "(48, 48, 3)", "num_classes": len(CLASS_NAMES), "total_params": 0}

# -------------------- Pages --------------------
@app.route("/")
def index():
    facts = get_model_facts()
    history_exists = os.path.exists(METRICS_IMG)
    return render_template("index.html", facts=facts, history_exists=history_exists,
                           CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD)
    
@app.route("/upload", methods=["GET"])
def upload():
    if model is None:
        flash("Model not loaded. Check server logs.")
    return render_template("upload.html", back_url=url_for("index"))


@app.route("/realtime")
def realtime():
    # Page must contain the canvas-overlay WebRTC client (realtime.html below)
    return render_template("realtime.html", back_url=url_for("index"),
                           CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("Model not loaded. Check server logs.")
        return redirect(url_for("index"))

    if "image" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload an image of a proper fromat.")
        return redirect(url_for("upload"))

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        flash("Could not read the uploaded image.")
        return redirect(url_for("index"))

    batch = preprocess_bgr(img_bgr)
    preds = model.predict(batch, verbose=0)
    class_id = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
    passed = conf >= CONFIDENCE_THRESHOLD

    return render_template(
        "result.html",
        image_url=url_for("static", filename=f"uploads/{filename}"),
        label=label,
        class_id=class_id,
        confidence=f"{conf*100:.2f}%",
        passed=passed,
        threshold=int(CONFIDENCE_THRESHOLD * 100),
        back_url=url_for("index")
    )

@app.route("/about")
def about():
    facts = get_model_facts()
    return render_template("about.html", facts=facts, back_url=url_for("index"))
 
 
@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# -------------------- WebRTC (aiortc) — reader-based pipeline --------------------
active_peers = set()

def _safe_send_json(dc, payload):
    try:
        if dc and dc.readyState == "open":
            dc.send(json.dumps(payload, separators=(",", ":")))
    except Exception:
        pass

async def _frame_reader(track, dc, model, class_names, img_size):
    """
    Consume frames from the incoming WebRTC video track, run inference,
    and send JSON results over the datachannel.
    """
    # Optional warmup for consistent latency
    # _ = model.predict(np.zeros((1, img_size[1], img_size[0], 3), dtype=np.float32), verbose=0)

    last_time = time.time()
    frames = 0
    fps = 0.0

    while True:
        frame = await track.recv()  # drives the pipeline

        try:
            img = frame.to_ndarray(format="bgr24")
            # preprocess
            img_resized = cv2.resize(img, img_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype("float32") / 255.0
            batch = np.expand_dims(img_norm, axis=0)

            # predict
            preds = model.predict(batch, verbose=0)
            class_id = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0]))
            label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"

            # fps
            frames += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frames / (now - last_time)
                frames = 0
                last_time = now

            # send to browser
            _safe_send_json(dc, {
                "type": "prediction",
                "label": label,
                "class_id": class_id,
                "confidence": f"{conf*100:.2f}%",
                "fps": f"{fps:.1f}"
            })
            # Debug (optional):
            # print(f"Predicted: {label} {conf*100:.1f}% FPS:{fps:.1f}")

        except Exception:
            # ignore frame-level errors
            pass

@app.route("/webrtc/offer", methods=["POST"])
def webrtc_offer():
    if model is None:
        return ("Model not loaded", 500, {"Content-Type": "text/plain"})

    offer_sdp = request.get_data(as_text=True)
    offer = RTCSessionDescription(sdp=offer_sdp, type="offer")

    pc = RTCPeerConnection()
    active_peers.add(pc)

    holder = {"dc": None, "reader_future": None, "track": None}

    @pc.on("connectionstatechange")
    def on_conn_state_change():
        st = pc.connectionState
        if st in ("failed", "closed", "disconnected"):
            try:
                active_peers.discard(pc)
                # cancel reader if running
                fut = holder.get("reader_future")
                if fut and not fut.done():
                    fut.cancel()
            except Exception:
                pass
            # close pc on the global loop
            asyncio.run_coroutine_threadsafe(pc.close(), GLOBAL_LOOP)

    @pc.on("datachannel")
    def on_datachannel(channel):
        holder["dc"] = channel
        if holder.get("track") is not None and holder.get("reader_future") is None:
            holder["reader_future"] = asyncio.run_coroutine_threadsafe(
                _frame_reader(holder["track"], holder["dc"], model, CLASS_NAMES, IMG_SIZE),
                GLOBAL_LOOP
            )

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            holder["track"] = track
            if holder.get("dc") is not None and holder.get("reader_future") is None:
                holder["reader_future"] = asyncio.run_coroutine_threadsafe(
                    _frame_reader(holder["track"], holder["dc"], model, CLASS_NAMES, IMG_SIZE),
                    GLOBAL_LOOP
                )

    async def process():
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription.sdp

    # Run the SDP coroutine on the global loop and wait for its result here
    fut = asyncio.run_coroutine_threadsafe(process(), GLOBAL_LOOP)
    sdp_answer = fut.result(timeout=10)

    return (sdp_answer, 200, {"Content-Type": "application/sdp"})


# -------------------- Run --------------------
if __name__ == "__main__":
    # Open http://127.0.0.1:5000 for About, and /realtime for WebRTC view
    app.run(debug=True)
    