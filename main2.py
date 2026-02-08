import os
import json
import uuid
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, send_file
)

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ==== LOCAL IMPORTS ====
from model import (
    UNetDenoiser, DDPM,
    EfficientNetClassifier,
    get_eval_transform, get_eval_transform_ddpm
)

from doctor_db import DOCTORS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib.units import inch



# ===================================
# CONFIG
# ===================================
app = Flask(__name__)
app.secret_key = "oralytics-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==== MODEL FILES ====
CLF_PATH = os.path.join(BASE_DIR, "clf_out", "classifier_best.pth")
CLF_CLASSES = os.path.join(BASE_DIR, "clf_out", "class_names.json")
DDPM_PATH = os.path.join(BASE_DIR, "ddpm_out", "ddpm_final.pth")

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===================================
# CLINICAL RULE ENGINE
# ===================================

CLINICAL_RULES = {
    "non_cancerous": {
        "title": "Non-cancerous (Benign) Lesion",
        "summary": (
            "The image shows features that are more consistent with a benign lesion. "
            "These lesions usually have well-defined borders, slower growth, and "
            "less aggressive surface changes compared to cancerous lesions."
        ),
        "possible_conditions": [
            "Traumatic ulcer / frictional keratosis",
            "Benign hyperkeratosis",
            "Inflammatory or infectious lesion"
        ],
        "precautions": [
            "Maintain excellent oral hygiene (soft brushing, alcohol-free mouthwash).",
            "Avoid hot, spicy, or very acidic foods that irritate the area.",
            "Stop any local trauma (sharp teeth, biting habit, ill-fitting dentures).",
            "Avoid tobacco and alcohol completely.",
            "Return for review if the lesion does not heal within 10–14 days."
        ],
        "follow_up": (
            "Review in 2–3 weeks. If the lesion persists, shows rapid change, "
            "or symptoms worsen, clinical re-evaluation and possible biopsy are indicated."
        ),
        "urgency": "low"
    },

    "precancerous": {
        "title": "Potentially Malignant (Pre-cancerous) Lesion",
        "summary": (
            "The lesion shows features suggestive of a potentially malignant disorder. "
            "These lesions require careful monitoring and often histopathological "
            "confirmation to assess the degree of epithelial dysplasia."
        ),
        "possible_conditions": [
            "Leukoplakia / erythroleukoplakia",
            "Oral submucous fibrosis related changes",
            "Lichen planus with dysplastic areas"
        ],
        "precautions": [
            "Complete cessation of all tobacco products (smokeless and smoked).",
            "Avoid alcohol consumption to reduce mucosal irritation.",
            "Protect the lesion from repeated trauma or biting.",
            "Follow a balanced diet rich in fruits and vegetables (antioxidant support).",
            "Do not apply over-the-counter creams, gels, or home remedies on the lesion."
        ],
        "follow_up": (
            "Early specialist referral is recommended. Lesions of this nature should be "
            "clinically monitored with photographic documentation and, where indicated, "
            "biopsy for histopathological grading."
        ),
        "urgency": "moderate"
    },

    "cancerous": {
        "title": "Suspicious for Oral Cancer",
        "summary": (
            "The lesion shows several features that may be associated with malignant change, "
            "such as irregular margins, surface ulceration, color variation, or induration. "
            "This output is only a screening aid and must be confirmed by clinical "
            "examination and biopsy."
        ),
        "possible_conditions": [
            "Oral squamous cell carcinoma",
            "Invasive or ulceroproliferative malignant lesion",
            "Severe epithelial dysplasia / carcinoma in situ"
        ],
        "precautions": [
            "Immediate avoidance of all tobacco and alcohol products.",
            "Avoid spicy, very hot, or abrasive foods that traumatize the lesion.",
            "Do not attempt self-treatment or cauterization of the area.",
            "Inform the patient about the importance of early diagnosis.",
            "Encourage support from family for timely hospital visit."
        ],
        "follow_up": (
            "Urgent referral to an oral medicine/oral oncology specialist is recommended. "
            "Biopsy and staging investigations should be planned at the earliest possible time."
        ),
        "urgency": "high"
    }
}


def build_clinical_info(label: str, confidence: float) -> dict:
    """
    Given predicted label and confidence, return a structured explanation
    and recommendations from CLINICAL_RULES.
    """
    info = CLINICAL_RULES.get(label, CLINICAL_RULES["non_cancerous"]).copy()
    info["label"] = label
    info["confidence"] = confidence
    return info


# ===================================
# LOAD MODELS
# ===================================
classifier = None
unet = None
ddpm = None
class_names = None


def load_models():
    """Load classifier + DDPM models."""
    global classifier, unet, ddpm, class_names

    print("🔄 Loading models...")

    # ---- class names ----
    if not os.path.exists(CLF_CLASSES):
        default = ["cancerous", "non_cancerous", "precancerous"]
        with open(CLF_CLASSES, "w") as f:
            json.dump(default, f)

    with open(CLF_CLASSES, "r") as f:
        class_names = json.load(f)

    # ---- classifier ----
    if not os.path.exists(CLF_PATH):
        raise FileNotFoundError(f"Classifier model not found at: {CLF_PATH}")

    clf = EfficientNetClassifier(num_classes=len(class_names))
    clf.load_state_dict(torch.load(CLF_PATH, map_location=device))
    clf.to(device)
    clf.eval()
    classifier = clf

    # ---- DDPM ----
    if not os.path.exists(DDPM_PATH):
        raise FileNotFoundError(f"DDPM model not found at: {DDPM_PATH}")

    un = UNetDenoiser().to(device)
    un.load_state_dict(torch.load(DDPM_PATH, map_location=device))
    un.eval()

    global ddpm
    ddpm = DDPM(device=device)
    global unet
    unet = un

    print("✅ Models loaded (Classifier + DDPM).")


# ===================================
# ROUTES (UI PAGES)
# ===================================

@app.route("/")
def splash():
    return render_template("splash.html")


@app.route("/welcome")
def welcome():
    return render_template("welcome.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ==== LOGIN ====
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        doc = DOCTORS.get(u)

        if doc and doc["password"] == p:
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
    if "doctor" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


# ===================================
# PREDICTION ROUTE
# ===================================
@app.route("/predict", methods=["POST"])
def predict():
    if "doctor" not in session:
        return redirect(url_for("login"))

    # ---- patient form data ----
    session["name"] = request.form.get("name", "")
    session["age"] = request.form.get("age", "")
    session["gender"] = request.form.get("gender", "")
    session["duration"] = request.form.get("duration", "")
    session["symptoms"] = request.form.get("symptoms", "")
    session["habits"] = request.form.get("habits", "")

    # ---- image upload ----
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    unique_name = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(save_path)

    session["uploaded_original"] = unique_name

    # =====================================
    # 1. CLASSIFIER INFERENCE
    # =====================================
    img = Image.open(save_path).convert("RGB")

    # classifier transform (ImageNet-like)
    clf_transform = get_eval_transform(image_size=224)
    clf_tensor = clf_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = classifier(clf_tensor)
        probs = F.softmax(out, dim=1)

    conf, idx = torch.max(probs, dim=1)
    confidence = float(conf.item())
    pred_label = class_names[idx.item()]  # 'cancerous' / 'non_cancerous' / 'precancerous'

    # =====================================
    # 2. DDPM: NOISY + DENOISED VISUALIZATION
    # =====================================
    ddpm_tf = get_eval_transform_ddpm(image_size=64)  # [-1,1]
    ddpm_tensor = ddpm_tf(img).unsqueeze(0).to(device)  # [1,3,64,64]

    t_step = 50
    t = torch.full((1,), t_step, device=device, dtype=torch.long)
    noise = torch.randn_like(ddpm_tensor)
    x_t = ddpm.q_sample(ddpm_tensor, t, noise)  # noisy image

    # Reverse diffusion for visualization
    den = ddpm.p_sample_from(unet, x_t.clone(), start_t=t_step, steps=40)

    # Convert [-1,1] tensors back to [0,1] and then to PIL
    from torchvision.transforms.functional import to_pil_image

    def minus1_1_to_pil(tensor_4d):
        # tensor_4d: [1,3,H,W]
        t = tensor_4d.squeeze(0).detach().cpu().clamp(-1, 1)
        t = (t + 1) / 2.0  # -> [0,1]
        return to_pil_image(t)

    noisy_pil = minus1_1_to_pil(x_t)
    den_pil = minus1_1_to_pil(den)

    noisy_name = f"noisy_{unique_name}"
    den_name = f"denoised_{unique_name}"

    noisy_path = os.path.join(UPLOAD_DIR, noisy_name)
    den_path = os.path.join(UPLOAD_DIR, den_name)

    noisy_pil.save(noisy_path)
    den_pil.save(den_path)

    # =====================================
    # 3. CLINICAL INTERPRETATION (RULE ENGINE)
    # =====================================
    clinical_info = build_clinical_info(pred_label, confidence)

    # Store in session for PDF/report
    session["prediction"] = pred_label
    session["confidence"] = confidence
    session["diag_title"] = clinical_info["title"]
    session["diag_summary"] = clinical_info["summary"]
    session["diag_possible_conditions"] = clinical_info["possible_conditions"]
    session["diag_precautions"] = clinical_info["precautions"]
    session["diag_follow_up"] = clinical_info["follow_up"]
    session["diag_urgency"] = clinical_info["urgency"]

    # =====================================
    # 4. RENDER RESULT PAGE
    # =====================================
    return render_template(
        "result.html",
        # images
        original_url=url_for("static", filename="uploads/" + unique_name),
        noisy_url=url_for("static", filename="uploads/" + noisy_name),
        denoised_url=url_for("static", filename="uploads/" + den_name),
        # prediction
        prediction=pred_label,
        confidence=confidence,
        # clinical info
        diag_title=clinical_info["title"],
        diag_summary=clinical_info["summary"],
        possible_conditions=clinical_info["possible_conditions"],
        precautions=clinical_info["precautions"],
        follow_up=clinical_info["follow_up"],
        urgency=clinical_info["urgency"]
    )


# ===================================
# PDF EXPORT
# ===================================
@app.route("/download_pdf")
def download_pdf():
    try:
        filename = session.get("uploaded_original")
        if not filename:
            return "No report data available", 400

        pred = session.get("prediction", "N/A")
        conf = session.get("confidence", 0.0)

        diag_summary = session.get(
            "diag_summary",
            "Clinical evaluation indicates benign oral lesion with no immediate malignant features detected."
        )

        precautions = session.get(
            "precautions",
            "Maintain oral hygiene. Avoid tobacco and alcohol. Schedule routine follow-up. Seek immediate care if symptoms worsen."
        )

        doctor = DOCTORS.get(session.get("doctor"), {})

        pdf_path = os.path.join(REPORT_DIR, filename.replace(".png", ".pdf"))

        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

        margin_x = 50
        margin_y = 60
        current_y = height - 120

        styles = getSampleStyleSheet()
        normal = styles["Normal"]

        # ========= PAGE TEMPLATE =========
        def draw_page_template():
            # Watermark
            logo_path = os.path.join(BASE_DIR, "static", "images", "oralytics-modified.png")
            if os.path.exists(logo_path):
                c.saveState()
                c.setFillAlpha(0.07)
                c.drawImage(
                    logo_path,
                    width/2 - 160,
                    height/2 - 160,
                    width=320,
                    height=320,
                    preserveAspectRatio=True,
                    mask='auto'
                )
                c.restoreState()

            # Header
            c.setFont("Helvetica-Bold", 19)
            c.drawCentredString(width/2, height - 60, "ORAL LESION DIAGNOSTIC REPORT")

            c.setFont("Helvetica", 11)
            c.drawCentredString(width/2, height - 85, "AI-Assisted Clinical Decision Support System")

            c.line(margin_x, height - 95, width - margin_x, height - 95)

        def check_page_space(y_needed=120):
            nonlocal current_y
            if current_y < y_needed:
                c.showPage()
                draw_page_template()
                current_y = height - 120

        # Draw first page
        draw_page_template()

        # ========= SECTION FUNCTION =========
        def draw_section(title, content):
            nonlocal current_y
            check_page_space(140)

            c.setFont("Helvetica-Bold", 13)
            c.drawString(margin_x, current_y, title)
            current_y -= 18

            para = Paragraph(content, normal)
            w, h = para.wrap(width - 2*margin_x, current_y - margin_y)

            check_page_space(h + 40)

            para.drawOn(c, margin_x, current_y - h)
            current_y -= h + 30

        # ========= PATIENT DETAILS =========
        draw_section(
            "Patient Information",
            f"""
            <b>Name:</b> {session.get('name','')}<br/>
            <b>Age:</b> {session.get('age','')}<br/>
            <b>Gender:</b> {session.get('gender','')}
            """
        )

        # ========= DIAGNOSIS =========
        draw_section(
            "Diagnosis Result",
            f"""
            <b>Predicted Class:</b> {pred.replace('_',' ').title()}<br/>
            <b>Confidence Score:</b> {round(conf*100,2)} %
            """
        )

        # ========= CLINICAL INTERPRETATION =========
        draw_section(
            "Clinical Interpretation",
            diag_summary
        )

        # ========= PRECAUTIONS =========
        precaution_points = "<br/>".join(
            ["• " + p.strip() for p in precautions.split(".") if p.strip()]
        )

        draw_section(
            "Recommended Precautions",
            precaution_points
        )

        # ========= DOCTOR =========
        draw_section(
            "Diagnosed By",
            f"""
            <b>{doctor.get('name','')}</b><br/>
            {doctor.get('qualification','')}<br/>
            Email: {doctor.get('email','')}<br/>
            Phone: {doctor.get('phone','')}
            """
        )

        # ========= FOOTER =========
        c.setFont("Helvetica-Oblique", 9)
        c.drawCentredString(
            width/2,
            45,
            "This AI-generated report is intended for clinical decision support only."
        )
        c.drawCentredString(
            width/2,
            30,
            "Oralytics AI © 2025 | Confidential Medical Record"
        )

        c.save()
        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        return f"PDF generation error: {str(e)}", 500


#===================================
# MAIN
# ===================================
if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
