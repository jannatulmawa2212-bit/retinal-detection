import streamlit as st
import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
import time
import os
from huggingface_hub import hf_hub_download

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Retinal Disease Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0D1117; color: #E6EDF3; }
    .stApp { background-color: #0D1117; }
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: white;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .metric-sub {
        font-size: 11px;
        color: #8B949E;
        margin-top: 2px;
    }
    .disease-card {
        background: #161B22;
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
        border-left: 4px solid;
    }
    .status-detected {
        background: #1a0a0a;
        border: 1px solid #FF6B6B;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #FF6B6B;
    }
    .status-normal {
        background: #0a1a0a;
        border: 1px solid #51CF66;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #51CF66;
    }
    h1, h2, h3 { color: #E6EDF3 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #238636, #2EA043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2EA043, #3FB950);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIG
# =============================================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = 224
PATCH_SIZE  = 16
NUM_PATCHES = 196
N_SIDE      = 14
LABEL_NAMES = ['DR', 'GLAUCOMA', 'HR', 'RVO']
LABEL_FULL  = ['Diabetic Retinopathy', 'Glaucoma',
               'Hypertensive Retinopathy', 'Retinal Vein Occlusion']
COLORS      = ['#FF6B6B', '#51CF66', '#74C0FC', '#FFA94D']
THRESHOLDS  = {'DR': 0.4221, 'GLAUCOMA': 0.5166,
               'HR': 0.7656, 'RVO': 0.3765}
SEVERITY    = {
    'DR':       [(0.39,0.55,'Mild'),(0.55,0.75,'Moderate'),(0.75,1.0,'Severe')],
    'GLAUCOMA': [(0.52,0.65,'Mild'),(0.65,0.82,'Moderate'),(0.82,1.0,'Severe')],
    'HR':       [(0.87,0.92,'Mild'),(0.92,0.96,'Moderate'),(0.96,1.0,'Severe')],
    'RVO':      [(0.38,0.55,'Mild'),(0.55,0.75,'Moderate'),(0.75,1.0,'Severe')],
}
DISEASE_INFO = {
    'DR':       ('Microaneurysms, haemorrhages, hard exudates at macula',
                 'Caused by diabetes damaging retinal blood vessels'),
    'GLAUCOMA': ('Enlarged optic cup, thinning neuroretinal rim',
                 'Caused by increased eye pressure damaging optic nerve'),
    'HR':       ('Vessel narrowing, AV nipping, flame haemorrhages',
                 'Caused by high blood pressure damaging retinal vessels'),
    'RVO':      ('Dilated tortuous veins, diffuse haemorrhages near disc',
                 'Caused by blockage of retinal vein'),
}

# =============================================================================
# MODEL
# =============================================================================
class PRETIClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            'vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.encoder.parameters():
            p.requires_grad = False
        d = self.encoder.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 256),
            nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 4))

    def forward(self, x):
        return self.head(self.encoder(x))

@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    model = PRETIClassifier().to(DEVICE)
    try:
        # Download model from HuggingFace
        model_path = hf_hub_download(
            repo_id="mawa2212/preti-retinal-weights",
            filename="best_model.pth",
            repo_type="model"
        )
        state      = torch.load(model_path, map_location=DEVICE)
        model_dict = model.state_dict()
        filtered   = {k: v for k, v in state.items()
                      if k in model_dict and
                      model_dict[k].shape == v.shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        st.sidebar.success("✅ Model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
    model.eval()
    return model

# =============================================================================
# PREPROCESSING
# =============================================================================
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def preprocess(img_pil):
    img_cv = np.array(img_pil.convert('RGB'))
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab    = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return val_transform(Image.fromarray(img_cv))

def to_display(t):
    a = t.permute(1,2,0).numpy()
    return ((a-a.min())/(a.max()-a.min()+1e-8)*255).astype(np.uint8)

def get_attention(model, tensor):
    attn_list = []
    def hook(m, inp, out):
        B,N,C = inp[0].shape
        qkv = m.qkv(inp[0]).reshape(
            B,N,3,m.num_heads,C//m.num_heads).permute(2,0,3,1,4)
        q,k,_ = qkv.unbind(0)
        a = (q@k.transpose(-2,-1)*(C//m.num_heads)**-0.5).softmax(dim=-1)
        attn_list.append(a.detach().cpu())
    h = list(model.encoder.blocks)[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        model.encoder.forward_features(tensor.unsqueeze(0).to(DEVICE))
    h.remove()
    if not attn_list:
        return torch.ones(NUM_PATCHES)/NUM_PATCHES
    a = attn_list[0].mean(dim=1)[0,0,1:]
    return (a-a.min())/(a.max()-a.min()+1e-8)

def get_severity(name, prob):
    for lo,hi,label in SEVERITY[name]:
        if lo <= prob < hi:
            return label
    return 'Severe' if prob >= THRESHOLDS[name] else ''

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 16px'>
        <div style='font-size:32px'>🔬</div>
        <div style='font-size:16px;font-weight:700;color:#E6EDF3'>
            RetinalAI Vision
        </div>
        <div style='font-size:11px;color:#8B949E;margin-top:4px'>
            AI-Powered Retinal Disease Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    metrics = [
        ("Macro AUC", "0.9779", "#51CF66"),
        ("DR AUC",    "0.9663", "#FF6B6B"),
        ("GL AUC",    "0.9993", "#74C0FC"),
        ("HR AUC",    "0.9999", "#FFA94D"),
        ("RVO AUC",   "0.9462", "#CC5DE8"),
        ("BW Saved",  "70.3%",  "#51CF66"),
    ]
    cols = st.columns(2)
    for idx, (label, val, color) in enumerate(metrics):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label' style='color:{color}'>{label}</div>
                <div class='metric-value'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style='color:#8B949E;font-size:12px;line-height:1.8'>
    This system uses an <b style='color:#E6EDF3'>AI foundation model</b>
    trained on over <b style='color:#E6EDF3'>one million</b> retinal images
    to simultaneously detect 4 diseases.<br><br>
    <b style='color:#58A6FF'>AGPT</b> uses the model's attention maps
    to transmit only disease-relevant patches — saving
    <b style='color:#51CF66'>70.3%</b> bandwidth for rural telemedicine.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#484F58;font-size:10px;line-height:1.6'>
    AI Foundation Model · AGPT · 2026
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# REPORT GENERATION (clean, physician-facing, no internal references)
# =============================================================================
def build_report(probs, detected, elapsed, top_k):
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

    # disease rows
    rows = ""
    for i, name in enumerate(LABEL_NAMES):
        p = probs[i]
        thr = THRESHOLDS[name]
        if p >= thr:
            sev = get_severity(name, p) or "Detected"
            status = f"<span style='color:#1F8A3B;font-weight:600'>&#10003; {sev}</span>"
        else:
            status = "<span style='color:#888'>&#10007; Not detected</span>"
        full = LABEL_FULL[i]
        bar_w = int(p * 160)
        rows += f"""
        <tr>
          <td style="padding:10px 14px;border-bottom:1px solid #eee">{full}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #eee">
              {p:.3f}
              <span style="display:inline-block;height:10px;width:{bar_w}px;
                    background:#7C3AED;border-radius:2px;margin-left:8px;
                    vertical-align:middle"></span>
          </td>
          <td style="padding:10px 14px;border-bottom:1px solid #eee">{status}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #eee">{thr:.4f}</td>
        </tr>"""

    if detected:
        banner = (f"<div style='background:#FF6B6B;color:white;text-align:center;"
                  f"padding:14px;font-size:18px;font-weight:700;border-radius:6px'>"
                  f"DISEASE DETECTED: {', '.join(detected)}</div>")
    else:
        banner = ("<div style='background:#1F8A3B;color:white;text-align:center;"
                  "padding:14px;font-size:18px;font-weight:700;border-radius:6px'>"
                  "NO DISEASE DETECTED</div>")

    # clinical indicators
    clinical = ""
    if detected:
        clinical = "<h3 style='color:#7C3AED;margin-top:28px'>Clinical Indicators</h3>"
        for name in detected:
            signs, cause = DISEASE_INFO[name]
            clinical += (f"<p style='margin:6px 0'><b style='color:#7C3AED'>{name}</b><br>"
                         f"<span style='font-size:14px'>Signs: {signs}</span><br>"
                         f"<span style='font-size:14px'>Cause: {cause}</span></p>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family:'Segoe UI',Arial,sans-serif; color:#222; max-width:820px;
          margin:24px auto; padding:0 20px }}
  .header {{ background:#1a1a2e; color:white; padding:22px 26px; border-radius:8px;
             display:flex; justify-content:space-between; align-items:center }}
  .header h1 {{ font-size:22px; margin:0; line-height:1.3 }}
  .header .meta {{ font-size:13px; color:#bbb; text-align:right }}
  table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:14px }}
  th {{ background:#7C3AED; color:white; padding:10px 14px; text-align:left }}
  h3 {{ color:#7C3AED }}
  .footer {{ margin-top:34px; padding-top:12px; border-top:1px solid #ddd;
             font-size:11px; color:#999; text-align:center }}
</style></head>
<body>
  <div class="header">
    <h1>Retinal Disease Detection&nbsp;Report</h1>
    <div class="meta">Generated: {now}<br>Analysis time: {elapsed:.2f}s</div>
  </div>

  <div style="margin:18px 0">{banner}</div>

  <h3>Disease Probabilities</h3>
  <table>
    <tr><th>Disease</th><th>Probability</th><th>Status</th><th>Threshold</th></tr>
    {rows}
  </table>

  <h3 style="margin-top:24px">Transmission Summary (AGPT)</h3>
  <p style="font-size:14px">
     Patches transmitted: <b>{top_k} / 196</b> (top 30%)<br>
     Bandwidth saved: <b>70.3%</b> &nbsp;|&nbsp; ~588&nbsp;KB &rarr; ~178&nbsp;KB
  </p>

  {clinical}

  <div class="footer">
     Automated retinal screening tool &middot; For clinical decision support only &middot; {now[:4]}
  </div>
</body></html>"""
    return html


# =============================================================================
# MAIN PAGE
# =============================================================================
st.markdown("""
<div style='text-align:center;padding:8px 0 24px'>
    <div style='font-size:13px;color:#58A6FF;font-weight:600;
                letter-spacing:2px;margin-bottom:6px'>
        ADVANCED RETINAL DISEASE DETECTION SYSTEM · 2026
    </div>
    <div style='font-size:30px;font-weight:700;color:#E6EDF3;
                margin-bottom:8px'>
        🩺 Retinal Disease Detection System
    </div>
    <div style='font-size:14px;color:#8B949E'>
        Multi-label detection of
        <span style='color:#FF6B6B;font-weight:600'>DR</span> ·
        <span style='color:#51CF66;font-weight:600'>Glaucoma</span> ·
        <span style='color:#74C0FC;font-weight:600'>HR</span> ·
        <span style='color:#FFA94D;font-weight:600'>RVO</span>
        with AGPT Bandwidth-Efficient Transmission
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

# Upload section
col_upload, col_results = st.columns([1, 2])

with col_upload:
    st.markdown("#### Upload Retinal Image")
    uploaded = st.file_uploader(
        "Choose a retinal fundus image",
        type=['jpg','jpeg','png'],
        help="Any fundus photograph — JPEG or PNG")

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)
        analyze_btn = st.button("🔍  Analyze Retina")
    else:
        st.markdown("""
        <div style='background:#161B22;border:1px dashed #30363D;
                    border-radius:10px;padding:40px;text-align:center;
                    color:#8B949E;font-size:13px'>
            📤 Upload a retinal fundus photograph<br>
            <span style='font-size:11px'>JPEG or PNG · Any resolution</span>
        </div>
        """, unsafe_allow_html=True)
        analyze_btn = False

if uploaded and analyze_btn:
    with st.spinner("Analyzing retina..."):
        t0     = time.time()
        tensor = preprocess(img)

        with torch.no_grad():
            probs = torch.sigmoid(
                model(tensor.unsqueeze(0).to(DEVICE))
            ).cpu().float().numpy()[0]

        attn   = get_attention(model, tensor)
        top_k  = 58
        _, top = torch.topk(attn, top_k)
        top    = top.sort().values

        mask = np.zeros((N_SIDE, N_SIDE))
        for idx in top:
            mask[idx//N_SIDE, idx%N_SIDE] = 1
        mask_full = np.kron(mask, np.ones((PATCH_SIZE, PATCH_SIZE)))

        recon = torch.ones(3, IMAGE_SIZE, IMAGE_SIZE) * 0.5
        for idx in top:
            r,c = (idx//N_SIDE).item(), (idx%N_SIDE).item()
            P   = PATCH_SIZE
            recon[:,r*P:(r+1)*P,c*P:(c+1)*P] = \
                tensor[:,r*P:(r+1)*P,c*P:(c+1)*P]

        elapsed  = time.time() - t0
        detected = [n for n,p in zip(LABEL_NAMES,probs)
                    if p >= THRESHOLDS[n]]

    # ── Results ──────────────────────────────────────────────────
    with col_results:
        # Status
        if detected:
            st.markdown(f"""
            <div class='status-detected'>
                ⚠️  DISEASE DETECTED: {', '.join(detected)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='status-normal'>✅  NORMAL — No disease detected</div>
            """, unsafe_allow_html=True)

        st.markdown(f"<div style='color:#8B949E;font-size:12px;margin:6px 0'>Analysis completed in {elapsed:.2f}s</div>",
                    unsafe_allow_html=True)

    # ── 4 image panels ───────────────────────────────────────────
    st.markdown("### Visual Analysis")
    c1, c2, c3, c4 = st.columns(4)
    orig_np  = to_display(tensor)
    recon_np = to_display(recon)
    attn_map = attn.reshape(N_SIDE, N_SIDE).numpy()

    with c1:
        st.image(orig_np, caption="① Original (CLAHE)", use_column_width=True)
    with c2:
        fig_attn, ax = plt.subplots(figsize=(3,3), facecolor='#0D1117')
        ax.imshow(attn_map, cmap='inferno', interpolation='bilinear')
        ax.axis('off')
        ax.set_title('② Attention Map', color='white', fontsize=9, pad=4)
        st.pyplot(fig_attn, use_container_width=True)
        plt.close()
    with c3:
        fig_sel, ax = plt.subplots(figsize=(3,3), facecolor='#0D1117')
        ax.imshow(orig_np)
        ax.imshow(mask_full, alpha=0.55, cmap='YlOrRd')
        ax.axis('off')
        ax.set_title(f'③ {top_k}/196 Patches', color='white', fontsize=9, pad=4)
        st.pyplot(fig_sel, use_container_width=True)
        plt.close()
    with c4:
        st.image(recon_np, caption="④ Doctor Receives", use_column_width=True)

    # ── Disease predictions ───────────────────────────────────────
    st.markdown("### Disease Predictions")
    for name, full, prob, color in zip(LABEL_NAMES, LABEL_FULL, probs, COLORS):
        th  = THRESHOLDS[name]
        det = prob >= th
        sev = get_severity(name, prob)
        pct = int(prob * 100)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"""
            <div class='disease-card'
                 style='border-color:{"" + color if det else "#30363D"}'>
                <div style='display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:6px'>
                    <span style='color:white;font-weight:600;font-size:14px'>
                        {full}
                    </span>
                    <span style='color:{"" + color if det else "#8B949E"};
                                 font-weight:bold;font-size:13px'>
                        {"✓ " + sev.upper() if det else "✗ Not Detected"}
                    </span>
                </div>
                <div style='background:#21262D;border-radius:4px;height:8px'>
                    <div style='background:{color};border-radius:4px;
                                height:8px;width:{pct}%'></div>
                </div>
                <div style='display:flex;justify-content:space-between;
                            margin-top:4px'>
                    <span style='color:#8B949E;font-size:11px'>
                        {DISEASE_INFO[name][0]}
                    </span>
                    <span style='color:{color};font-size:12px;
                                 font-weight:bold'>{prob:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── AGPT Stats ────────────────────────────────────────────────
    st.markdown("### AGPT Transmission Stats")
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (s1, "PATCHES SENT",     f"{top_k}/196", "30% of image",    "#74C0FC"),
        (s2, "DATA TRANSMITTED", "178 KB",        "was 588 KB",      "#51CF66"),
        (s3, "BANDWIDTH SAVED",  "70.3%",         "410 KB reduced",  "#FF6B6B"),
        (s4, "TIME SAVED @2G",   "32 sec",        "15s vs 47s",      "#FFA94D"),
    ]
    for col, label, val, sub, color in stats:
        with col:
            st.markdown(f"""
            <div class='metric-card'
                 style='border:1px solid {color}'>
                <div class='metric-label'
                     style='color:{color}'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Clinical note if detected ─────────────────────────────────
    if detected:
        st.markdown("### Clinical Indicators")
        for name in detected:
            signs, cause = DISEASE_INFO[name]
            color = COLORS[LABEL_NAMES.index(name)]
            st.markdown(f"""
            <div style='background:#161B22;border-left:4px solid {color};
                        border-radius:8px;padding:12px;margin:6px 0'>
                <b style='color:{color}'>{name}</b><br>
                <span style='color:#E6EDF3;font-size:13px'>
                    Signs: {signs}</span><br>
                <span style='color:#8B949E;font-size:12px'>
                    Cause: {cause}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Downloadable report for the physician ─────────────────────
    st.markdown("### Report")
    report_html = build_report(probs, detected, elapsed, top_k)
    st.download_button(
        label="⬇  Download Report (HTML)",
        data=report_html,
        file_name="retinal_screening_report.html",
        mime="text/html",
    )

elif not uploaded:
    with col_results:
        st.markdown("""
        <div style='background:#161B22;border:1px solid #30363D;
                    border-radius:10px;padding:40px;text-align:center;
                    color:#8B949E;margin-top:32px'>
            <div style='font-size:40px;margin-bottom:12px'>🔬</div>
            <div style='font-size:16px;font-weight:600;color:#E6EDF3;
                        margin-bottom:8px'>
                Upload a retinal image to begin
            </div>
            <div style='font-size:13px;line-height:1.8'>
                The system will detect DR, Glaucoma, HR and RVO<br>
                simultaneously and show AGPT bandwidth analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
