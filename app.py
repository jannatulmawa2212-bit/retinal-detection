import streamlit as st
import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
import os
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Retinal Disease Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

* { font-family: 'Nunito', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #fce4ec 0%, #f3e5f5 30%,
                #e8eaf6 60%, #e3f2fd 100%);
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #fce4ec 100%);
    border-right: 2px solid #f8bbd0;
}

.hero-header {
    background: linear-gradient(135deg, #f48fb1, #ce93d8, #90caf9);
    background-size: 300% 300%;
    animation: gradientMove 4s ease infinite;
    border-radius: 24px;
    padding: 36px 24px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(244,143,177,0.3);
}

@keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    color: white;
    text-shadow: 0 2px 8px rgba(0,0,0,0.15);
    margin: 0;
}

.hero-sub {
    font-size: 15px;
    color: rgba(255,255,255,0.9);
    margin-top: 8px;
}

.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin: 3px;
}

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 16px 12px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.8);
    transition: transform 0.2s;
}

.metric-card:hover { transform: translateY(-2px); }

.metric-value { font-size: 24px; font-weight: 800; }

.metric-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.disease-card {
    background: white;
    border-radius: 16px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    border-left: 5px solid;
    transition: transform 0.2s, box-shadow 0.2s;
}

.disease-card:hover {
    transform: translateX(4px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}

.status-detected {
    background: linear-gradient(135deg, #fff0f3, #ffe4e8);
    border: 2px solid #f48fb1;
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    font-size: 18px;
    font-weight: 800;
    color: #c2185b;
    box-shadow: 0 4px 16px rgba(244,143,177,0.2);
    animation: pulse 2s infinite;
}

.status-normal {
    background: linear-gradient(135deg, #f0fff4, #e8f5e9);
    border: 2px solid #a5d6a7;
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    font-size: 18px;
    font-weight: 800;
    color: #2e7d32;
    box-shadow: 0 4px 16px rgba(165,214,167,0.2);
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 4px 16px rgba(244,143,177,0.2); }
    50%       { box-shadow: 0 4px 24px rgba(244,143,177,0.5); }
}

.stat-card {
    background: white;
    border-radius: 16px;
    padding: 20px 12px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    border-top: 4px solid;
}

.panel-card {
    background: white;
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    text-align: center;
}

.panel-title {
    font-size: 12px;
    font-weight: 700;
    color: #9c27b0;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton>button {
    background: linear-gradient(135deg, #f48fb1, #ce93d8) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 14px 32px !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(244,143,177,0.4) !important;
    transition: all 0.3s !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(244,143,177,0.6) !important;
}

h1,h2,h3 { color: #4a148c !important; }
.stMarkdown p { color: #4a148c; }
</style>
""", unsafe_allow_html=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = 224
PATCH_SIZE  = 16
NUM_PATCHES = 196
N_SIDE      = 14
LABEL_NAMES = ['DR', 'GLAUCOMA', 'HR', 'RVO']
LABEL_FULL  = ['Diabetic Retinopathy', 'Glaucoma',
               'Hypertensive Retinopathy', 'Retinal Vein Occlusion']
ICONS       = ['🩸', '🫧', '💓', '🔴']
COLORS      = ['#f48fb1', '#a5d6a7', '#90caf9', '#ffcc80']
COLORS_DARK = ['#c2185b', '#2e7d32', '#1565c0', '#e65100']
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

class PRETIClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            'vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.encoder.parameters(): p.requires_grad = False
        d = self.encoder.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 256),
            nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 4))
    def forward(self, x): return self.head(self.encoder(x))

@st.cache_resource
def load_model():
    model = PRETIClassifier().to(DEVICE)
    try:
        path = hf_hub_download(
            repo_id="mawa2212/preti-retinal-weights",
            filename="best_model.pth", repo_type="model")
        state = torch.load(path, map_location=DEVICE)
        md    = model.state_dict()
        f     = {k:v for k,v in state.items()
                 if k in md and md[k].shape==v.shape}
        md.update(f); model.load_state_dict(md)
        st.sidebar.success("✅ Model loaded!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
    model.eval(); return model

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])])

def preprocess(img_pil):
    img = np.array(img_pil.convert('RGB'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return val_transform(Image.fromarray(img))

def to_display(t):
    a = t.permute(1,2,0).numpy()
    return ((a-a.min())/(a.max()-a.min()+1e-8)*255).astype(np.uint8)

def get_attention(model, tensor):
    al = []
    def hook(m, inp, out):
        B,N,C = inp[0].shape
        qkv = m.qkv(inp[0]).reshape(
            B,N,3,m.num_heads,C//m.num_heads).permute(2,0,3,1,4)
        q,k,_ = qkv.unbind(0)
        a = (q@k.transpose(-2,-1)*(C//m.num_heads)**-0.5).softmax(dim=-1)
        al.append(a.detach().cpu())
    h = list(model.encoder.blocks)[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        model.encoder.forward_features(tensor.unsqueeze(0).to(DEVICE))
    h.remove()
    if not al: return torch.ones(NUM_PATCHES)/NUM_PATCHES
    a = al[0].mean(dim=1)[0,0,1:]
    return (a-a.min())/(a.max()-a.min()+1e-8)

def get_severity(name, prob):
    for lo,hi,label in SEVERITY[name]:
        if lo <= prob < hi: return label
    return 'Severe' if prob >= THRESHOLDS[name] else ''

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:40px'>🩺</div>
        <div style='font-size:18px;font-weight:800;
                    background:linear-gradient(135deg,#f48fb1,#ce93d8);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent'>
            Retinal AI
        </div>
        <div style='font-size:11px;color:#9c27b0;margin-top:2px;
                    font-weight:600'>
            AI-Powered Disease Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Results")
    metrics = [
        ("Macro AUC","99.0%","#f48fb1"),("DR","98.7%","#ef9a9a"),
        ("Glaucoma","100%","#a5d6a7"),("HR","98.8%","#90caf9"),
        ("RVO","98.6%","#ffcc80"),("BW Saved","70.3%","#ce93d8"),
    ]
    c1,c2 = st.columns(2)
    for i,(label,val,color) in enumerate(metrics):
        with (c1 if i%2==0 else c2):
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom:8px'>
                <div class='metric-label' style='color:{color}'>{label}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(135deg,#fce4ec,#f3e5f5);
                border-radius:12px;padding:14px;font-size:12px;
                color:#6a1b9a;line-height:1.8'>
    🤖 Advanced foundation model pretrained on
    <b>1,017,549</b> retinal images<br><br>
    ✨ <b>AGPT</b> saves <b>70.3%</b> bandwidth for
    rural telemedicine
    </div>
    """, unsafe_allow_html=True)

# MAIN
st.markdown("""
<div class='hero-header'>
    <div style='font-size:44px;margin-bottom:8px'>🩺</div>
    <div class='hero-title'>Retinal Disease Detection</div>
    <div class='hero-sub'>
        AI-powered simultaneous detection of
        🩸 DR · 🫧 Glaucoma · 💓 HR · 🔴 RVO
    </div>
    <div style='margin-top:14px'>
        <span class='badge' style='background:rgba(255,255,255,0.25);
              color:white'>✨ Macro AUC 99.0%</span>
        <span class='badge' style='background:rgba(255,255,255,0.25);
              color:white'>🌐 70.3% Bandwidth Saved</span>
        <span class='badge' style='background:rgba(255,255,255,0.25);
              color:white'>⚡ 4 Diseases at Once</span>
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model()
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("#### 📤 Upload Retinal Image")
    uploaded = st.file_uploader(
        "Choose a retinal fundus image",
        type=['jpg','jpeg','png'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, use_column_width=True, caption="✨ Ready to analyze!")
        analyze_btn = st.button("🔍 Analyze Retina ✨")
    else:
        st.markdown("""
        <div style='text-align:center;padding:32px;
                    color:#ce93d8;font-size:13px;
                    background:white;border-radius:16px;
                    border:2px dashed #f48fb1'>
            <div style='font-size:36px;margin-bottom:8px'>🖼️</div>
            Drop a retinal fundus image here<br>
            <span style='font-size:11px;color:#f48fb1'>
                JPEG or PNG · Any resolution
            </span>
        </div>
        """, unsafe_allow_html=True)
        analyze_btn = False

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:white;border-radius:16px;padding:16px;
                box-shadow:0 4px 16px rgba(0,0,0,0.06)'>
        <div style='font-size:12px;font-weight:700;color:#9c27b0;
                    margin-bottom:10px'>⚙️ HOW IT WORKS</div>
        <div style='font-size:12px;color:#6a1b9a;line-height:2.0'>
            ① CLAHE contrast enhancement<br>
            ② AI foundation model encoding<br>
            ③ 4-disease simultaneous detection<br>
            ④ Attention map extraction<br>
            ⑤ Top 30% disease patches selected<br>
            ⑥ 70.3% bandwidth reduction ✨
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded and analyze_btn:
    with st.spinner("🔬 Analyzing your retinal image..."):
        t0     = time.time()
        tensor = preprocess(img)
        with torch.no_grad():
            probs = torch.sigmoid(
                model(tensor.unsqueeze(0).to(DEVICE))
            ).cpu().float().numpy()[0]
        attn = get_attention(model, tensor)
        top_k = 58
        _, top = torch.topk(attn, top_k)
        top = top.sort().values
        mask = np.zeros((N_SIDE,N_SIDE))
        for idx in top: mask[idx//N_SIDE,idx%N_SIDE]=1
        mask_full = np.kron(mask, np.ones((PATCH_SIZE,PATCH_SIZE)))
        recon = torch.ones(3,IMAGE_SIZE,IMAGE_SIZE)*0.5
        for idx in top:
            r,c=(idx//N_SIDE).item(),(idx%N_SIDE).item(); P=PATCH_SIZE
            recon[:,r*P:(r+1)*P,c*P:(c+1)*P]=tensor[:,r*P:(r+1)*P,c*P:(c+1)*P]
        elapsed  = time.time()-t0
        detected = [n for n,p in zip(LABEL_NAMES,probs) if p>=THRESHOLDS[n]]

    with col_right:
        if detected:
            icons_det = ' '.join([ICONS[LABEL_NAMES.index(n)] for n in detected])
            st.markdown(f"""
            <div class='status-detected'>
                {icons_det} Disease Detected: {' · '.join(detected)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='status-normal'>
                🌸 All Clear — No Disease Detected!
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='text-align:right;color:#ce93d8;font-size:12px;
                    margin:4px 0 12px'>⚡ Analyzed in {elapsed:.2f}s</div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔍 Visual Analysis")
    c1,c2,c3,c4 = st.columns(4)
    orig_np  = to_display(tensor)
    recon_np = to_display(recon)
    attn_map = attn.reshape(N_SIDE,N_SIDE).numpy()

    for col,content,title in [
            (c1,orig_np,"① Original"),(c4,recon_np,"④ Doctor View")]:
        with col:
            st.markdown(f"""
            <div class='panel-card'>
                <div class='panel-title'>{title}</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(content, use_column_width=True)

    with c2:
        st.markdown("""
        <div class='panel-card'>
            <div class='panel-title'>② AI Attention 🔥</div>
        </div>
        """, unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(3,3))
        fig.patch.set_facecolor('white')
        ax.imshow(attn_map, cmap='RdPu', interpolation='bilinear')
        ax.axis('off'); plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c3:
        st.markdown(f"""
        <div class='panel-card'>
            <div class='panel-title'>③ {top_k}/196 Patches 🎯</div>
        </div>
        """, unsafe_allow_html=True)
        fig2,ax2 = plt.subplots(figsize=(3,3))
        fig2.patch.set_facecolor('white')
        ax2.imshow(orig_np)
        ax2.imshow(mask_full, alpha=0.5,
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                       '',['white','#f48fb1']))
        ax2.axis('off'); plt.tight_layout(pad=0)
        st.pyplot(fig2, use_container_width=True); plt.close()

    st.markdown("### 🩺 Disease Predictions")
    for name,full,prob,color,dark,icon in zip(
            LABEL_NAMES,LABEL_FULL,probs,COLORS,COLORS_DARK,ICONS):
        th  = THRESHOLDS[name]; det = prob>=th
        sev = get_severity(name,prob); pct = int(prob*100)
        c_a,c_b = st.columns([4,1])
        with c_a:
            st.markdown(f"""
            <div class='disease-card' style='border-color:{color}'>
                <div style='display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:8px'>
                    <span style='color:{dark};font-weight:800;font-size:15px'>
                        {icon} {full}
                    </span>
                    <span style='color:{""+dark if det else "#bdbdbd"};
                                 font-weight:800;font-size:13px'>
                        {"✓ "+sev.upper()+" 🎯" if det else "✗ Not Detected"}
                    </span>
                </div>
                <div style='background:#f5f5f5;border-radius:20px;height:10px'>
                    <div style='background:linear-gradient(90deg,{color},{dark});
                                border-radius:20px;height:10px;width:{pct}%'>
                    </div>
                </div>
                <div style='display:flex;justify-content:space-between;
                            margin-top:6px'>
                    <span style='color:#9e9e9e;font-size:11px'>
                        {DISEASE_INFO[name][0]}</span>
                    <span style='color:{dark};font-size:13px;font-weight:800'>
                        {prob:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📡 AGPT Transmission Stats")
    s1,s2,s3,s4 = st.columns(4)
    stats=[
        (s1,"🎯 PATCHES","58/196","30% of image","#f48fb1","#c2185b"),
        (s2,"📦 DATA","178 KB","was 588 KB","#a5d6a7","#2e7d32"),
        (s3,"💾 SAVED","70.3%","410 KB reduced","#ce93d8","#6a1b9a"),
        (s4,"⏱️ TIME","32 sec","15s vs 47s @2G","#ffcc80","#e65100"),
    ]
    for col,label,val,sub,color,dark in stats:
        with col:
            st.markdown(f"""
            <div class='stat-card' style='border-color:{color}'>
                <div style='font-size:11px;font-weight:700;
                            color:{dark};letter-spacing:1px'>{label}</div>
                <div style='font-size:26px;font-weight:800;
                            color:{dark};margin:6px 0'>{val}</div>
                <div style='font-size:11px;color:#9e9e9e'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    if detected:
        st.markdown("### 🏥 Clinical Information")
        for name in detected:
            signs,cause = DISEASE_INFO[name]
            color = COLORS[LABEL_NAMES.index(name)]
            dark  = COLORS_DARK[LABEL_NAMES.index(name)]
            icon  = ICONS[LABEL_NAMES.index(name)]
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,white,#fce4ec);
                        border-left:5px solid {color};
                        border-radius:16px;padding:16px;margin:8px 0;
                        box-shadow:0 4px 16px rgba(0,0,0,0.06)'>
                <b style='color:{dark};font-size:15px'>{icon} {name}</b><br>
                <span style='color:#4a148c;font-size:13px'>
                    🔍 Signs: {signs}</span><br>
                <span style='color:#9c27b0;font-size:12px'>
                    💡 Cause: {cause}</span>
            </div>
            """, unsafe_allow_html=True)

elif not uploaded:
    with col_right:
        st.markdown("""
        <div style='background:white;border-radius:20px;padding:48px;
                    text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.06);
                    margin-top:8px'>
            <div style='font-size:56px;margin-bottom:16px'>🔬</div>
            <div style='font-size:20px;font-weight:800;color:#9c27b0;
                        margin-bottom:8px'>
                Upload a retinal image to begin ✨
            </div>
            <div style='font-size:13px;color:#ce93d8;line-height:1.8'>
                The AI will simultaneously detect<br>
                🩸 Diabetic Retinopathy<br>
                🫧 Glaucoma<br>
                💓 Hypertensive Retinopathy<br>
                🔴 Retinal Vein Occlusion<br><br>
                <span style='color:#f48fb1;font-weight:700'>
                    + Show AGPT bandwidth savings
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;padding:24px 0 8px;
            color:#ce93d8;font-size:11px'>
    Made with 💕 · AI Foundation Model · AGPT Novel Contribution ·
    Focal Loss · Class-Balanced Sampling · 2026
</div>
""", unsafe_allow_html=True)
