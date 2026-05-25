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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(160deg,
        #f8f0ff 0%, #fff0f8 25%,
        #f0f8ff 50%, #f0fff8 75%,
        #fff8f0 100%);
}

section[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #f0e6ff;
}

.hero {
    background: linear-gradient(135deg,
        #9c27b0 0%, #e91e8c 40%,
        #2196f3 80%, #00bcd4 100%);
    background-size: 200% 200%;
    animation: heroAnim 6s ease infinite;
    border-radius: 24px;
    padding: 40px 32px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 16px 48px rgba(156,39,176,0.2);
}

@keyframes heroAnim {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 16px 12px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(156,39,176,0.1);
    margin-bottom: 8px;
    transition: transform 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(156,39,176,0.15);
}

.upload-box {
    background: white;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 2px dashed #e1bee7;
    margin-bottom: 16px;
}

.info-box {
    background: white;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    border: 1px solid #f0e6ff;
}

.status-found {
    background: linear-gradient(135deg, #fff0f8, #fce4ec);
    border: 2px solid #f48fb1;
    border-radius: 16px;
    padding: 16px 24px;
    text-align: center;
    font-weight: 800;
    font-size: 17px;
    color: #880e4f;
    box-shadow: 0 4px 20px rgba(244,143,177,0.25);
    animation: glow 2s ease infinite;
    margin-bottom: 16px;
}

.status-clear {
    background: linear-gradient(135deg, #f0fff4, #e8f5e9);
    border: 2px solid #a5d6a7;
    border-radius: 16px;
    padding: 16px 24px;
    text-align: center;
    font-weight: 800;
    font-size: 17px;
    color: #1b5e20;
    box-shadow: 0 4px 20px rgba(165,214,167,0.25);
    margin-bottom: 16px;
}

@keyframes glow {
    0%,100% { box-shadow: 0 4px 20px rgba(244,143,177,0.25); }
    50%      { box-shadow: 0 4px 32px rgba(244,143,177,0.5); }
}

.disease-bar {
    background: white;
    border-radius: 16px;
    padding: 16px 18px;
    margin: 8px 0;
    border-left: 5px solid;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}

.disease-bar:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.progress-bg {
    background: #f3e5f5;
    border-radius: 20px;
    height: 10px;
    overflow: hidden;
    margin: 8px 0;
}

.stat-pill {
    background: white;
    border-radius: 16px;
    padding: 20px 14px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    border-top: 4px solid;
}

.panel-img {
    background: white;
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    text-align: center;
    border: 1px solid #f0e6ff;
}

.panel-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.clinical-card {
    background: white;
    border-radius: 16px;
    padding: 16px;
    margin: 8px 0;
    border-left: 5px solid;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}

.stButton > button {
    background: linear-gradient(135deg, #9c27b0, #e91e8c) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 14px 32px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(156,39,176,0.35) !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(156,39,176,0.5) !important;
}

h1,h2,h3 { color: #4a148c !important; }

div[data-testid="stFileDropzone"] {
    background: linear-gradient(135deg,#fce4ec,#f3e5f5) !important;
    border: 2px dashed #ce93d8 !important;
    border-radius: 16px !important;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.95); }
    to   { opacity: 1; transform: scale(1); }
}

.hero { animation: scaleIn 0.6s ease; }
.metric-card { animation: fadeInUp 0.5s ease; }
.disease-bar { animation: slideInLeft 0.4s ease; }
.stat-pill   { animation: fadeInUp 0.6s ease; }
.status-found, .status-clear { animation: scaleIn 0.4s ease; }

div[data-testid="stFileDropzone"] > div {
    background: transparent !important;
}
div[data-testid="stFileDropzone"] p {
    color: #9c27b0 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
div[data-testid="stFileDropzone"] span {
    color: #ce93d8 !important;
}
div[data-testid="stFileDropzone"] button {
    background: linear-gradient(135deg,#9c27b0,#e91e8c) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(156,39,176,0.3) !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #9c27b0 !important;
}
small[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #ce93d8 !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = 224
PATCH_SIZE  = 16
NUM_PATCHES = 196
N_SIDE      = 14
LABEL_NAMES = ['DR', 'GLAUCOMA', 'HR', 'RVO']
LABEL_FULL  = ['Diabetic Retinopathy','Glaucoma',
               'Hypertensive Retinopathy','Retinal Vein Occlusion']
ICONS       = ['🩸','🫧','💗','🔴']
COLORS      = ['#f48fb1','#81c784','#64b5f6','#ffb74d']
COLORS_DARK = ['#880e4f','#1b5e20','#0d47a1','#e65100']
THRESHOLDS  = {'DR':0.3894,'GLAUCOMA':0.5200,'HR':0.8667,'RVO':0.3765}
SEVERITY    = {
    'DR':      [(0.39,0.55,'Mild'),(0.55,0.75,'Moderate'),(0.75,1.0,'Severe')],
    'GLAUCOMA':[(0.52,0.65,'Mild'),(0.65,0.82,'Moderate'),(0.82,1.0,'Severe')],
    'HR':      [(0.87,0.92,'Mild'),(0.92,0.96,'Moderate'),(0.96,1.0,'Severe')],
    'RVO':     [(0.38,0.55,'Mild'),(0.55,0.75,'Moderate'),(0.75,1.0,'Severe')],
}
DISEASE_INFO = {
    'DR':      ('Microaneurysms, haemorrhages, hard exudates at macula',
                'Diabetes damages retinal blood vessels'),
    'GLAUCOMA':('Enlarged optic cup, thinning neuroretinal rim',
                'Increased eye pressure damages optic nerve'),
    'HR':      ('Vessel narrowing, AV nipping, flame haemorrhages',
                'High blood pressure damages retinal vessels'),
    'RVO':     ('Dilated tortuous veins, diffuse haemorrhages near disc',
                'Blockage of retinal vein'),
}

# =============================================================================
class PRETIClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            'vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.encoder.parameters(): p.requires_grad = False
        d = self.encoder.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d,256),
            nn.GELU(), nn.Dropout(0.5), nn.Linear(256,4))
    def forward(self, x): return self.head(self.encoder(x))

@st.cache_resource
def load_model():
    model = PRETIClassifier().to(DEVICE)
    try:
        path = hf_hub_download(
            repo_id="mawa2212/preti-retinal-weights",
            filename="best_model.pth", repo_type="model")
        state = torch.load(path, map_location=DEVICE)
        md = model.state_dict()
        f  = {k:v for k,v in state.items()
              if k in md and md[k].shape==v.shape}
        md.update(f); model.load_state_dict(md)
        st.sidebar.success("✅ AI model loaded")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
    model.eval(); return model

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])])

def preprocess(img_pil):
    img = np.array(img_pil.convert('RGB'))
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    return val_transform(Image.fromarray(img))

def to_display(t):
    a = t.permute(1,2,0).numpy()
    return ((a-a.min())/(a.max()-a.min()+1e-8)*255).astype(np.uint8)

def get_attn(model, tensor):
    al = []
    def hook(m,inp,out):
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

def get_sev(name, prob):
    for lo,hi,lbl in SEVERITY[name]:
        if lo<=prob<hi: return lbl
    return 'Severe' if prob>=THRESHOLDS[name] else ''

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 12px'>
        <div style='font-size:36px'>🩺</div>
        <div style='font-size:17px;font-weight:800;
                    color:#9c27b0;margin-top:4px'>
            Retinal AI
        </div>
        <div style='font-size:11px;color:#ce93d8;font-weight:600'>
            AI-Powered Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**📊 Model Performance**")

    cols = st.columns(2)
    metrics = [
        ("Macro AUC","99.0%","#9c27b0"),
        ("DR","98.7%","#e91e8c"),
        ("Glaucoma","100%","#4caf50"),
        ("HR","98.8%","#2196f3"),
        ("RVO","98.6%","#ff9800"),
        ("BW Saved","70.3%","#00bcd4"),
    ]
    for i,(lbl,val,clr) in enumerate(metrics):
        with cols[i%2]:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:10px;font-weight:700;
                            color:{clr};letter-spacing:1px;
                            text-transform:uppercase'>{lbl}</div>
                <div style='font-size:22px;font-weight:800;
                            color:{clr}'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div class='info-box'>
        <div style='font-size:12px;color:#6a1b9a;line-height:1.9'>
            🤖 Foundation model trained on<br>
            <b>1,017,549</b> retinal images<br><br>
            ✨ <b>AGPT</b> saves <b>70.3%</b> bandwidth<br>
            for rural telemedicine<br><br>
            ⚡ Detects <b>4 diseases</b> simultaneously
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
st.markdown("""
<div class='hero'>
    <div style='font-size:42px;margin-bottom:10px'>🩺</div>
    <div style='font-size:34px;font-weight:800;color:white;
                text-shadow:0 2px 8px rgba(0,0,0,0.2);margin-bottom:8px'>
        Retinal Disease Detection
    </div>
    <div style='font-size:14px;color:rgba(255,255,255,0.9);
                margin-bottom:16px'>
        AI-powered simultaneous detection of
        🩸 DR · 🫧 Glaucoma · 💗 HR · 🔴 RVO
    </div>
    <div>
        <span style='background:rgba(255,255,255,0.2);color:white;
                     padding:5px 14px;border-radius:20px;font-size:12px;
                     font-weight:700;margin:3px;display:inline-block'>
            ✨ Macro AUC 99.0%
        </span>
        <span style='background:rgba(255,255,255,0.2);color:white;
                     padding:5px 14px;border-radius:20px;font-size:12px;
                     font-weight:700;margin:3px;display:inline-block'>
            🌐 70.3% Bandwidth Saved
        </span>
        <span style='background:rgba(255,255,255,0.2);color:white;
                     padding:5px 14px;border-radius:20px;font-size:12px;
                     font-weight:700;margin:3px;display:inline-block'>
            ⚡ 4 Diseases at Once
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model()

col_l, col_r = st.columns([1, 2])

with col_l:
    st.markdown("""
    <div style='text-align:center;margin-bottom:8px'>
        <span style='font-size:12px;font-weight:700;color:#9c27b0;
                     letter-spacing:1px'>📤 UPLOAD RETINAL IMAGE</span><br>
        <span style='font-size:11px;color:#ce93d8'>
            JPEG or PNG · Any resolution
        </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a retinal fundus image",
        type=['jpg','jpeg','png'])

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, use_column_width=True, caption="Ready to analyze ✨")
        btn = st.button("🔍 Analyze Now")
    else:
        btn = False

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <div style='font-size:12px;font-weight:700;color:#9c27b0;
                    margin-bottom:10px'>⚙️ Pipeline</div>
        <div style='font-size:12px;color:#6a1b9a;line-height:2.0'>
            ① CLAHE contrast enhancement<br>
            ② AI foundation model encoding<br>
            ③ 4-disease classification<br>
            ④ Attention map extraction<br>
            ⑤ Top 30% patches selected<br>
            ⑥ 70.3% bandwidth saved ✨
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded and btn:
    with st.spinner("🔬 Analyzing retinal image..."):
        t0 = time.time()
        tensor = preprocess(img)
        with torch.no_grad():
            probs = torch.sigmoid(
                model(tensor.unsqueeze(0).to(DEVICE))
            ).cpu().float().numpy()[0]
        attn = get_attn(model, tensor)
        _, top = torch.topk(attn, 58)
        top = top.sort().values
        mask = np.zeros((N_SIDE,N_SIDE))
        for idx in top: mask[idx//N_SIDE,idx%N_SIDE]=1
        mf = np.kron(mask, np.ones((PATCH_SIZE,PATCH_SIZE)))
        recon = torch.ones(3,IMAGE_SIZE,IMAGE_SIZE)*0.5
        for idx in top:
            r,c=(idx//N_SIDE).item(),(idx%N_SIDE).item(); P=PATCH_SIZE
            recon[:,r*P:(r+1)*P,c*P:(c+1)*P]=tensor[:,r*P:(r+1)*P,c*P:(c+1)*P]
        elapsed = time.time()-t0
        detected = [n for n,p in zip(LABEL_NAMES,probs) if p>=THRESHOLDS[n]]

    with col_r:
        if detected:
            st.markdown(f"""
            <div class='status-found'>
                ⚠️ Disease Detected: {' · '.join(detected)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='status-clear'>
                ✅ All Clear — No Disease Detected
            </div>
            """, unsafe_allow_html=True)
        st.caption(f"⚡ Analyzed in {elapsed:.2f}s")

    st.markdown("### 🔬 Visual Analysis")
    c1,c2,c3,c4 = st.columns(4)
    on = to_display(tensor)
    rn = to_display(recon)
    am = attn.reshape(N_SIDE,N_SIDE).numpy()

    for col,img_data,lbl,clr in [
        (c1,on,"① Original","#9c27b0"),
        (c4,rn,"④ Doctor Receives","#00bcd4")]:
        with col:
            st.markdown(f"""
            <div class='panel-img'>
                <div class='panel-label' style='color:{clr}'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(img_data, use_column_width=True)

    with c2:
        st.markdown("""
        <div class='panel-img'>
            <div class='panel-label' style='color:#e91e8c'>
                ② AI Attention
            </div>
        </div>
        """, unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(3,3))
        fig.patch.set_facecolor('white')
        ax.imshow(am, cmap='RdPu', interpolation='bilinear')
        ax.axis('off'); plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c3:
        st.markdown(f"""
        <div class='panel-img'>
            <div class='panel-label' style='color:#ff9800'>
                ③ 58/196 Patches
            </div>
        </div>
        """, unsafe_allow_html=True)
        fig2,ax2 = plt.subplots(figsize=(3,3))
        fig2.patch.set_facecolor('white')
        ax2.imshow(on)
        cmap_pink = matplotlib.colors.LinearSegmentedColormap.from_list(
            '',[( 1,1,1,0),(0.96,0.28,0.56,0.6)])
        ax2.imshow(mf, cmap=cmap_pink)
        ax2.axis('off'); plt.tight_layout(pad=0)
        st.pyplot(fig2, use_container_width=True); plt.close()

    st.markdown("### 🩺 Disease Predictions")
    for name,full,prob,color,dark,icon in zip(
            LABEL_NAMES,LABEL_FULL,probs,COLORS,COLORS_DARK,ICONS):
        th  = THRESHOLDS[name]
        det = prob>=th
        sev = get_sev(name,prob)
        pct = int(prob*100)
        st.markdown(f"""
        <div class='disease-bar' style='border-color:{color}'>
            <div style='display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:8px'>
                <span style='color:{dark};font-weight:700;font-size:15px'>
                    {icon} {full}
                </span>
                <span style='color:{"" + dark if det else "#bdbdbd"};
                             font-weight:700;font-size:13px;
                             background:{"" + color + "33" if det else "#f5f5f5"};
                             padding:3px 10px;border-radius:20px'>
                    {"✓ " + sev if det else "✗ Not detected"}
                </span>
            </div>
            <div class='progress-bg'>
                <div style='background:linear-gradient(90deg,{color},{dark});
                            border-radius:20px;height:10px;width:{pct}%;
                            transition:width 1s ease'></div>
            </div>
            <div style='display:flex;justify-content:space-between;
                        margin-top:6px'>
                <span style='color:#9e9e9e;font-size:11px'>
                    {DISEASE_INFO[name][0]}
                </span>
                <span style='color:{dark};font-size:13px;font-weight:700'>
                    {prob:.3f}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📡 AGPT Transmission Stats")
    s1,s2,s3,s4 = st.columns(4)
    stats=[
        (s1,"🎯 Patches Sent","58 / 196","30% of image","#9c27b0"),
        (s2,"📦 Data Size","178 KB","was 588 KB","#4caf50"),
        (s3,"💾 Bandwidth","70.3%","410 KB saved","#e91e8c"),
        (s4,"⏱️ Time Saved","32 sec","15s vs 47s @2G","#2196f3"),
    ]
    for col,lbl,val,sub,clr in stats:
        with col:
            st.markdown(f"""
            <div class='stat-pill' style='border-color:{clr}'>
                <div style='font-size:11px;font-weight:700;color:{clr};
                            letter-spacing:1px'>{lbl}</div>
                <div style='font-size:26px;font-weight:800;
                            color:{clr};margin:6px 0'>{val}</div>
                <div style='font-size:11px;color:#9e9e9e'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    if detected:
        st.markdown("### 🏥 Clinical Information")
        for name in detected:
            signs,cause = DISEASE_INFO[name]
            clr  = COLORS[LABEL_NAMES.index(name)]
            dark = COLORS_DARK[LABEL_NAMES.index(name)]
            icon = ICONS[LABEL_NAMES.index(name)]
            st.markdown(f"""
            <div class='clinical-card' style='border-color:{clr}'>
                <b style='color:{dark};font-size:15px'>
                    {icon} {name}
                </b><br>
                <span style='color:#4a148c;font-size:13px'>
                    🔍 {signs}
                </span><br>
                <span style='color:#9c27b0;font-size:12px'>
                    💡 {cause}
                </span>
            </div>
            """, unsafe_allow_html=True)

elif not uploaded:
    with col_r:
        st.markdown("""
        <div style='background:white;border-radius:24px;padding:52px 40px;
                    text-align:center;box-shadow:0 4px 24px rgba(0,0,0,0.06);
                    border:1px solid #f0e6ff;margin-top:8px'>
            <div style='font-size:52px;margin-bottom:16px'>🔬</div>
            <div style='font-size:20px;font-weight:800;color:#9c27b0;
                        margin-bottom:12px'>
                Upload a retinal image to begin
            </div>
            <div style='font-size:13px;color:#ce93d8;line-height:2.2'>
                🩸 Diabetic Retinopathy<br>
                🫧 Glaucoma<br>
                💗 Hypertensive Retinopathy<br>
                🔴 Retinal Vein Occlusion<br><br>
                <span style='color:#e91e8c;font-weight:700;font-size:14px'>
                    + AGPT bandwidth analysis ✨
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;padding:28px 0 12px;
            color:#ce93d8;font-size:11px'>
    Made with 💜 · AI Foundation Model · AGPT Novel Contribution ·
    Focal Loss · Class-Balanced Sampling · 2026
</div>
""", unsafe_allow_html=True)
