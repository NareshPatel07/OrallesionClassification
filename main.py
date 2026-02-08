
import os
import json
import uuid
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, send_file, flash
)

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage
from PIL import Image

# local imports (your model.py and doctor_db.py must be in same folder)
from model import UNetDenoiser, DDPM, EfficientNetClassifier, get_eval_transform
from doctor_db import DOCTORS

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =========================
# CONFIG
# =========================
app = Flask(__name__)
app.secret_key = "oralytics-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model files (update if your folders differ)
CLF_DIR = os.path.join(BASE_DIR, "clf_out")
CLF_PATH = os.path.join(CLF_DIR, "classifier_best.pth")
CLF_CLASSES = os.path.join(CLF_DIR, "class_names.json")

DDPM_DIR = os.path.join(BASE_DIR, "ddpm_out")
DDPM_PATH = os.path.join(DDPM_DIR, "ddpm_final.pth")

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# global model variables
classifier = None
unet = None
ddpm = None
class_names = ["cancerous", "non_cancerous", "precancerous"]


def load_models():
    """
    Try to load classifier and ddpm/unet. If missing, do NOT crash the server.
    This prevents BuildError and undefined variables at template render time.
    """
    global classifier, unet, ddpm, class_names

    print("🔄 Loading models...")

    # class names
    if os.path.exists(CLF_CLASSES):
        try:
            with open(CLF_CLASSES, "r") as f:
                class_names = json.load(f)
            print(f"✓ Loaded class names: {class_names}")
        except Exception as e:
            print("⚠ Failed to read class_names.json, using defaults:", e)
    else:
        # create default file if not present
        try:
            os.makedirs(CLF_DIR, exist_ok=True)
            with open(CLF_CLASSES, "w") as f:
                json.dump(class_names, f)
            print("⚠ Created default class_names.json")
        except Exception as e:
            print("⚠ Could not create class_names.json:", e)

    # classifier
    if os.path.exists(CLF_PATH):
        try:
            classifier = EfficientNetClassifier(num_classes=len(class_names))
            classifier.load_state_dict(torch.load(CLF_PATH, map_location=device))
            classifier.to(device)
            classifier.eval()
            print(f"✓ Classifier loaded from {CLF_PATH}")
        except Exception as e:
            print("⚠ Failed to load classifier:", e)
            classifier = None
    else:
        print(f"⚠ Classifier file not found at {CLF_PATH} — classifier disabled.")
        classifier = None

    # ddpm + unet
    if os.path.exists(DDPM_PATH):
        try:
            unet = UNetDenoiser().to(device)
            unet.load_state_dict(torch.load(DDPM_PATH, map_location=device))
            unet.eval()
            ddpm = DDPM(device=device)
            print(f"✓ DDPM & UNet loaded from {DDPM_PATH}")
        except Exception as e:
            print("⚠ Failed to load DDPM/UNet:", e)
            unet = None
            ddpm = None
    else:
        print(f"⚠ DDPM model not found at {DDPM_PATH} — noising/denoising disabled.")
        unet = None
        ddpm = None

    print("🔚 model loading step finished.")


# =========================
# ROUTES
# =========================

@app.route("/")
def root():
    # many of your pages link to 'about' — about exists in templates now
    return render_template("welcome.html")


@app.route("/welcome")
def welcome():
    return render_template("welcome.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")
        doc = DOCTORS.get(u)
        if doc and doc.get("password") == p:
            session["doctor"] = u
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("welcome"))


@app.route("/index")
def index():
    # require login
    if "doctor" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


# =========================
# PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    # require login
    if "doctor" not in session:
        return redirect(url_for("login"))

    # collect patient info
    session["name"] = request.form.get("name", "")
    session["age"] = request.form.get("age", "")
    session["gender"] = request.form.get("gender", "")
    session["duration"] = request.form.get("duration", "")
    session["symptoms"] = request.form.get("symptoms", "")
    session["habits"] = request.form.get("habits", "")

    # upload
    file = request.files.get("file")
    if not file:
        flash("No file uploaded", "error")
        return redirect(url_for("index"))

    # save original
    orig_fname = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(UPLOAD_DIR, orig_fname)
    file.save(save_path)
    session["uploaded_original"] = orig_fname

    # default outputs (used if models missing)
    pred_label = "uncertain"
    confidence = 0.0

    # prepare image tensor
    try:
        pil = Image.open(save_path).convert("RGB")
        transform = get_eval_transform(224)
        tensor = transform(pil).unsqueeze(0).to(device)
    except Exception as e:
        print("⚠ Failed to load/transform image:", e)
        tensor = None

    # CLASSIFICATION (if classifier exists)
    if classifier is not None and tensor is not None:
        with torch.no_grad():
            logits = classifier(tensor)
            probs = F.softmax(logits, dim=1)
            conf_val, idx = torch.max(probs, dim=1)
            confidence = float(conf_val.item())
            try:
                pred_label = class_names[int(idx.item())]
            except Exception:
                pred_label = "uncertain"

        # thresholding for uncertain predictions
        THRESH = 0.55
        if confidence < THRESH:
            pred_label = "uncertain"
    else:
        print("⚠ Classifier missing — returning 'uncertain' prediction.")

    # SAVE INTO SESSION so download_pdf can read them
    session["prediction"] = pred_label
    session["confidence"] = confidence

    # ---------- Create noised and denoised images ----------

    noisy_fname = f"noise_{orig_fname}"
    noisy_path = os.path.join(UPLOAD_DIR, noisy_fname)

    denoised_fname = f"denoise_{orig_fname}"
    denoised_path = os.path.join(UPLOAD_DIR, denoised_fname)

    if ddpm is not None and unet is not None and tensor is not None:
        try:
            # forward q_sample to get a noisy version (visual)
            t_noisy = torch.tensor([min(50, ddpm.timesteps - 1)], device=device, dtype=torch.long)
            noise = torch.randn_like(tensor)
            x_t = ddpm.q_sample(tensor, t_noisy, noise)

            # save noisy
            noisy_img = (x_t.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
            ToPILImage()(noisy_img).save(noisy_path)

            # denoise (run several reverse steps)
            den = ddpm.p_sample_from(unet, x_t, start_t=int(t_noisy.item()), steps=min(40, int(t_noisy.item())))
            den_img = (den.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
            ToPILImage()(den_img).save(denoised_path)
        except Exception as e:
            print("⚠ Error creating noisy/denoised images:", e)
            # fallback: copy original into placeholders
            try:
                Image.open(save_path).save(noisy_path)
                Image.open(save_path).save(denoised_path)
            except Exception:
                pass
    else:
        # model not available -> use original as placeholder
        try:
            Image.open(save_path).save(noisy_path)
            Image.open(save_path).save(denoised_path)
        except Exception as e:
            print("⚠ Could not create placeholder noise/denoised images:", e)

    # prepare urls to pass to template
    original_url = url_for("static", filename=f"uploads/{orig_fname}")
    noisy_url = url_for("static", filename=f"uploads/{noisy_fname}")
    denoised_url = url_for("static", filename=f"uploads/{denoised_fname}")

    # Render result page with all needed variables (so Jinja won't raise UndefinedError)
    return render_template(
        "result.html",
        original_url=original_url,
        noisy_url=noisy_url,
        denoised_url=denoised_url,
        prediction=pred_label,
        confidence=confidence
    )


# =========================
# DOWNLOAD PDF
# =========================
@app.route("/download_pdf")
def download_pdf():
    filename = session.get("uploaded_original")
    if not filename:
        return "No report available", 400

    pred = session.get("prediction", "Unknown")
    conf = session.get("confidence", 0.0)

    doctor = DOCTORS.get(session.get("doctor"), {})
    pdf_fname = filename.replace(".png", ".pdf")
    pdf_path = os.path.join(REPORT_DIR, pdf_fname)

    try:
        c = canvas.Canvas(pdf_path, pagesize=A4)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, 800, "🦷 Oral Lesion Diagnosis Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, 770, f"Patient Name: {session.get('name','')}")
        c.drawString(50, 750, f"Age: {session.get('age','')}")
        c.drawString(50, 730, f"Gender: {session.get('gender','')}")
        c.drawString(50, 700, f"Prediction: {pred}")
        c.drawString(50, 680, f"Confidence: {round(conf*100,2)}%")

        if doctor:
            c.drawString(50, 640, "Diagnosed By:")
            c.drawString(50, 620, f"{doctor.get('name','') } ({doctor.get('qualification','')})")
            c.drawString(50, 600, f"Email: {doctor.get('email','')}")
            c.drawString(50, 580, f"Phone: {doctor.get('phone','')}")

        c.save()
    except Exception as e:
        print("⚠ Error creating PDF:", e)
        return "PDF generation failed", 500

    return send_file(pdf_path, as_attachment=True)


# =========================
# START
# =========================
if __name__ == "__main__":
    load_models()
    # debug=True useful locally; remove in production
    app.run(host="0.0.0.0", port=5000, debug=True)
