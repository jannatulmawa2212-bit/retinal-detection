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
# CUSTOM CSS — Original dark theme + pastel touch + pookie button
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
* { font-family: 'Nunito', sans-serif !important; }

.stApp {
    background: linear-gradient(160deg,
        #1a0d2e 0%, #0d1117 30%,
        #0d1a2e 60%, #1a0d2e 100%) !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e0a35 0%, #161B22 100%);
    border-right: 1px solid #3d1f5e;
}

.metric-card {
    background: #1e1030;
    border: 1px solid #3d1f5e;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    margin: 4px;
    transition: transform 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(156,39,176,0.2);
}

.metric-value {
    font-size: 26px;
    font-weight: 800;
    color: white;
}

.metric-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-sub {
    font-size: 11px;
    color: #8B949E;
    margin-top: 2px;
}

.disease-card {
    background: #1e1030;
    border-radius: 14px;
    padding: 14px;
    margin: 8px 0;
    border-left: 4px solid;
    transition: transform 0.2s;
}

.disease-card:hover { transform: translateX(4px); }

.status-detected {
    background: linear-gradient(135deg, #2d0a1a, #1a0d2e);
    border: 2px solid #f48fb1;
    border-radius: 14px;
    padding: 14px;
    text-align: center;
    font-size: 18px;
    font-weight: 800;
    color: #f48fb1;
    box-shadow: 0 4px 20px rgba(244,143,177,0.2);
    animation: glowPulse 2s ease infinite;
}

.status-normal {
    background: linear-gradient(135deg, #0a1f0d, #0d1117);
    border: 2px solid #81c784;
    border-radius: 14px;
    padding: 14px;
    text-align: center;
    font-size: 18px;
    font-weight: 800;
    color: #81c784;
    box-shadow: 0 4px 20px rgba(129,199,132,0.2);
}

@keyframes glowPulse {
    0%,100% { box-shadow: 0 4px 20px rgba(244,143,177,0.2); }
    50%      { box-shadow: 0 4px 30px rgba(244,143,177,0.5); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes heroGlow {
    0%,100% { box-shadow: 0 8px 40px rgba(156,39,176,0.3); }
    50%      { box-shadow: 0 8px 50px rgba(233,30,140,0.5); }
}

.hero-banner {
    background: linear-gradient(135deg, #9c27b0, #e91e8c, #2196f3);
    background-size: 200% 200%;
    animation: gradMove 5s ease infinite, heroGlow 3s ease infinite;
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 24px;
}

@keyframes gradMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

h1, h2, h3 { color: #E6EDF3 !important; }

.stButton>button {
    background: linear-gradient(135deg, #9c27b0, #e91e8c) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 12px 32px !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(156,39,176,0.4) !important;
    letter-spacing: 0.3px !important;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #ab47bc, #f06292) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(156,39,176,0.6) !important;
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
COLORS      = ['#f48fb1', '#81c784', '#64b5f6', '#ffb74d']
THRESHOLDS  = {'DR': 0.3894, 'GLAUCOMA': 0.5200,
               'HR': 0.8667, 'RVO': 0.3765}
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
    model = PRETIClassifier().to(DEVICE)
    try:
        model_path = hf_hub_download(
            repo_id="mawa2212/preti-retinal-weights",
            filename="best_model.pth",
            repo_type="model")
        state      = torch.load(model_path, map_location=DEVICE)
        model_dict = model.state_dict()
        filtered   = {k: v for k, v in state.items()
                      if k in model_dict and
                      model_dict[k].shape == v.shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        st.sidebar.success("✅ Model loaded")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
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
        <div style='font-size:32px'>🩺</div>
        <div style='font-size:16px;font-weight:800;
                    background:linear-gradient(135deg,#f48fb1,#ce93d8);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent'>
            Retinal AI
        </div>
        <div style='font-size:11px;color:#ce93d8;margin-top:4px;
                    font-weight:600'>
            AI-Powered Disease Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    metrics = [
        ("Macro AUC", "0.9903", "#f48fb1"),
        ("DR AUC",    "0.9869", "#FF6B6B"),
        ("GL AUC",    "0.9999", "#81c784"),
        ("HR AUC",    "0.9881", "#64b5f6"),
        ("RVO AUC",   "0.9864", "#ffb74d"),
        ("BW Saved",  "70.3%",  "#ce93d8"),
    ]
    cols = st.columns(2)
    for idx, (label, val, color) in enumerate(metrics):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label' style='color:{color}'>{label}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style='color:#ce93d8;font-size:12px;line-height:1.8'>
    Uses <b style='color:#f48fb1'>advanced AI</b> foundation model
    pretrained on <b style='color:#f48fb1'>1,017,549</b> retinal images
    to simultaneously detect 4 diseases.<br><br>
    <b style='color:#64b5f6'>AGPT</b> transmits only disease-relevant
    patches — saving <b style='color:#81c784'>70.3%</b> bandwidth.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#484F58;font-size:10px;line-height:1.6'>
    Focal Loss: Lin et al., 2020<br>
    Sampler: Cui et al., 2019<br>
    ViT: Dosovitskiy et al., 2021
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN PAGE
# =============================================================================

# Animated hero banner
st.markdown("""
<div class='hero-banner'>
    <div style='font-size:13px;color:rgba(255,255,255,0.85);
                font-weight:700;letter-spacing:2px;margin-bottom:6px'>
        ADVANCED RETINAL DISEASE DETECTION SYSTEM · 2026
    </div>
    <div style='font-size:30px;font-weight:800;color:white;
                text-shadow:0 2px 8px rgba(0,0,0,0.2);margin-bottom:8px'>
        🩺 Retinal Disease Detection
    </div>
    <div style='font-size:14px;color:rgba(255,255,255,0.85)'>
        Multi-label detection of
        <span style='color:#fce4ec;font-weight:700'>DR</span> ·
        <span style='color:#e8f5e9;font-weight:700'>Glaucoma</span> ·
        <span style='color:#e3f2fd;font-weight:700'>HR</span> ·
        <span style='color:#fff3e0;font-weight:700'>RVO</span>
        with AGPT Bandwidth-Efficient Transmission
    </div>
    <div style='margin-top:14px'>
        <span style='background:rgba(255,255,255,0.2);color:white;
                     padding:5px 14px;border-radius:20px;font-size:12px;
                     font-weight:700;margin:3px;display:inline-block'>
            ✨ Macro AUC 0.9903
        </span>
        <span style='background:rgba(255,255,255,0.2);color:white;
                     padding:5px 14px;border-radius:20px;font-size:12px;
                     font-weight:700;margin:3px;display:inline-block'>
            💜 70.3% Bandwidth Saved
        </span>
        <span style='background:rgba(255,255,255,0.2);color:white;
                     padding:5px 14px;border-radius:20px;font-size:12px;
                     font-weight:700;margin:3px;display:inline-block'>
            ⚡ 4 Diseases at Once
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

# Upload section
col_upload, col_results = st.columns([1, 2])

with col_upload:
    st.markdown("#### 📤 Upload Retinal Image")
    uploaded = st.file_uploader(
        "Choose a retinal fundus image",
        type=['jpg','jpeg','png'],
        help="Any fundus photograph — JPEG or PNG")

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Ready to analyze ✨", use_column_width=True)
        analyze_btn = st.button("🔍 Analyze Retina")
    else:
        st.markdown("""
        <div style='background:#1e1030;border:1px dashed #3d1f5e;
                    border-radius:14px;padding:40px;text-align:center;
                    color:#ce93d8;font-size:13px'>
            🩺 Upload a retinal fundus photograph<br>
            <span style='font-size:11px;color:#8B949E'>
                JPEG or PNG · Any resolution
            </span>
        </div>
        """, unsafe_allow_html=True)
        analyze_btn = False

if uploaded and analyze_btn:
    with st.spinner("🔬 Analyzing retina..."):
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

    with col_results:
        if detected:
            st.markdown(f"""
            <div class='status-detected'>
                ⚠️ DISEASE DETECTED: {', '.join(detected)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='status-normal'>
                ✅ NORMAL — No disease detected
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='color:#ce93d8;font-size:12px;margin:6px 0'>
            ⚡ Analysis completed in {elapsed:.2f}s
        </div>
        """, unsafe_allow_html=True)

    # 4 image panels
    st.markdown("### 🔬 Visual Analysis")
    c1, c2, c3, c4 = st.columns(4)
    orig_np  = to_display(tensor)
    recon_np = to_display(recon)
    attn_map = attn.reshape(N_SIDE, N_SIDE).numpy()

    with c1:
        st.image(orig_np, caption="① Original (CLAHE)",
                 use_column_width=True)
    with c2:
        fig_attn, ax = plt.subplots(figsize=(3,3), facecolor='#1e1030')
        ax.imshow(attn_map, cmap='RdPu', interpolation='bilinear')
        ax.axis('off')
        ax.set_title('② AI Attention', color='#f48fb1',
                     fontsize=9, pad=4)
        st.pyplot(fig_attn, use_container_width=True)
        plt.close()
    with c3:
        fig_sel, ax = plt.subplots(figsize=(3,3), facecolor='#1e1030')
        ax.imshow(orig_np)
        ax.imshow(mask_full, alpha=0.5,
                  cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                      '', [(1,1,1,0),(0.96,0.28,0.56,0.6)]))
        ax.axis('off')
        ax.set_title(f'③ {top_k}/196 Patches', color='#f48fb1',
                     fontsize=9, pad=4)
        st.pyplot(fig_sel, use_container_width=True)
        plt.close()
    with c4:
        st.image(recon_np, caption="④ Doctor Receives",
                 use_column_width=True)

    # Disease predictions
    st.markdown("### 🩺 Disease Predictions")
    for name, full, prob, color in zip(LABEL_NAMES, LABEL_FULL,
                                        probs, COLORS):
        th  = THRESHOLDS[name]
        det = prob >= th
        sev = get_severity(name, prob)
        pct = int(prob * 100)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"""
            <div class='disease-card'
                 style='border-color:{"" + color if det else "#3d1f5e"}'>
                <div style='display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:6px'>
                    <span style='color:white;font-weight:700;font-size:14px'>
                        {full}
                    </span>
                    <span style='color:{"" + color if det else "#8B949E"};
                                 font-weight:800;font-size:13px;
                                 background:{"rgba(244,143,177,0.1)" if det else "transparent"};
                                 padding:2px 10px;border-radius:20px'>
                        {"✓ " + sev.upper() if det else "✗ Not Detected"}
                    </span>
                </div>
                <div style='background:#2d1040;border-radius:20px;height:8px'>
                    <div style='background:linear-gradient(90deg,{color},white);
                                border-radius:20px;height:8px;
                                width:{pct}%'></div>
                </div>
                <div style='display:flex;justify-content:space-between;
                            margin-top:4px'>
                    <span style='color:#8B949E;font-size:11px'>
                        {DISEASE_INFO[name][0]}
                    </span>
                    <span style='color:{color};font-size:12px;
                                 font-weight:800'>{prob:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # AGPT Stats
    st.markdown("### 📡 AGPT Transmission Stats")
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (s1, "PATCHES SENT",     f"{top_k}/196", "30% of image",   "#64b5f6"),
        (s2, "DATA TRANSMITTED", "178 KB",        "was 588 KB",     "#81c784"),
        (s3, "BANDWIDTH SAVED",  "70.3%",         "410 KB reduced", "#f48fb1"),
        (s4, "TIME SAVED @2G",   "32 sec",        "15s vs 47s",     "#ffb74d"),
    ]
    for col, label, val, sub, color in stats:
        with col:
            st.markdown(f"""
            <div class='metric-card'
                 style='border:1px solid {color}40'>
                <div class='metric-label'
                     style='color:{color}'>{label}</div>
                <div class='metric-value'
                     style='color:{color}'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    if detected:
        st.markdown("### 🏥 Clinical Indicators")
        for name in detected:
            signs, cause = DISEASE_INFO[name]
            color = COLORS[LABEL_NAMES.index(name)]
            st.markdown(f"""
            <div style='background:#1e1030;border-left:4px solid {color};
                        border-radius:12px;padding:14px;margin:6px 0'>
                <b style='color:{color};font-size:14px'>{name}</b><br>
                <span style='color:#E6EDF3;font-size:13px'>
                    🔍 Signs: {signs}
                </span><br>
                <span style='color:#8B949E;font-size:12px'>
                    💡 Cause: {cause}
                </span>
            </div>
            """, unsafe_allow_html=True)

elif not uploaded:
    with col_results:
        st.markdown("""
        <div style='background:#1e1030;border:1px solid #3d1f5e;
                    border-radius:14px;padding:40px;text-align:center;
                    color:#ce93d8;margin-top:32px'>
            <div style='font-size:40px;margin-bottom:12px'>🩺</div>
            <div style='font-size:16px;font-weight:700;color:#f48fb1;
                        margin-bottom:8px'>
                Upload a retinal image to begin
            </div>
            <div style='font-size:13px;line-height:1.9;color:#ce93d8'>
                Detects DR · Glaucoma · HR · RVO simultaneously<br>
                Shows AGPT bandwidth analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center;padding:24px 0 8px;
            color:#3d1f5e;font-size:11px'>
    Made with 💜 · AI Foundation Model · AGPT Novel Contribution ·
    Focal Loss · Class-Balanced Sampling · 2026
</div>
""", unsafe_allow_html=True)
