"""
CNN page — Image-based skin-type classifier.

Uses a lightweight CNN-style MLP on 32x32 grayscale pixel features.
On first call it samples up to MAX_PER_CLASS images per class, trains the model,
and generates visualisations. Training takes ~30-60 s.
"""
import os, glob, io, base64, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
DATA_ROOT = os.path.join(BASE, '..', '..', 'skin type image dataset')
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
CLASSES   = ['combination', 'dry', 'normal', 'oily']
IMG_SIZE  = 32          # resize to 32×32 greyscale  →  1 024 features
MAX_PER_CLASS = 400     # cap to keep training fast

CLASS_COLORS = {
    'combination': '#6c63ff',
    'dry':         '#38bdf8',
    'normal':      '#00d4aa',
    'oily':        '#ff6b6b',
}


# ── Dataset loading ───────────────────────────────────────────────────────────
def _load_dataset():
    X, y = [], []
    label_map = {c: i for i, c in enumerate(CLASSES)}
    for cls in CLASSES:
        folder = os.path.join(TRAIN_DIR, cls)
        files  = glob.glob(os.path.join(folder, '*.jpg'))
        random.seed(42)
        files  = random.sample(files, min(MAX_PER_CLASS, len(files)))
        for fp in files:
            try:
                img = Image.open(fp).convert('L').resize((IMG_SIZE, IMG_SIZE))
                X.append(np.array(img).flatten() / 255.0)
                y.append(label_map[cls])
            except Exception:
                pass
    return np.array(X), np.array(y)


# ── Image pre-processing (single upload) ────────────────────────────────────
def preprocess_image(file_bytes: bytes):
    """Return flat normalised pixel vector + thumbnail base64."""
    img  = Image.open(io.BytesIO(file_bytes)).convert('L').resize((IMG_SIZE, IMG_SIZE))
    vec  = np.array(img).flatten() / 255.0
    # thumbnail for display (RGB, 224×224)
    thumb = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize((224, 224))
    buf   = io.BytesIO()
    thumb.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    thumb_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.read()).decode()
    return vec, thumb_b64


# ── Visualisations ───────────────────────────────────────────────────────────
BG = '#050510'

def _b64_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def _plot_sensitivity(model, scaler, x_raw, predicted_class, file_bytes):
    """
    Pixel-sensitivity map:  perturb each pixel, measure how much the
    predicted-class probability changes → draw as a heatmap on the image.
    """
    x_sc   = scaler.transform([x_raw])[0]
    base_p = model.predict_proba([x_sc])[0][CLASSES.index(predicted_class)]

    # Compute sensitivity per pixel (vectorised approach)
    n_feat  = len(x_raw)
    delta   = 0.15
    sens    = np.zeros(n_feat)
    X_pert  = np.tile(x_sc, (n_feat, 1))
    for i in range(n_feat):
        X_pert[i, i] += delta
    probs = model.predict_proba(scaler.inverse_transform(X_pert))[:, CLASSES.index(predicted_class)]
    sens  = np.abs(probs - base_p)
    sens_map = sens.reshape(IMG_SIZE, IMG_SIZE)

    # Original image
    orig = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    orig_arr = np.array(orig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    titles = ['Uploaded Image', 'Sensitivity Heatmap', 'Overlay']
    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis('off')

    axes[0].imshow(orig_arr)
    axes[0].set_title(titles[0], color='white', fontsize=10, fontweight='600', pad=8)

    hm = axes[1].imshow(sens_map, cmap='hot', interpolation='bilinear')
    axes[1].set_title(titles[1], color='white', fontsize=10, fontweight='600', pad=8)
    cbar = fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#6272a4')
    cbar.outline.set_edgecolor('#2a2a55')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#6272a4', fontsize=7)

    # Overlay: blend heatmap on original
    sens_norm = (sens_map - sens_map.min()) / (sens_map.max() - sens_map.min() + 1e-9)
    axes[2].imshow(orig_arr, alpha=0.55)
    axes[2].imshow(sens_norm, cmap='hot', alpha=0.55, interpolation='bilinear')
    axes[2].set_title(titles[2], color='white', fontsize=10, fontweight='600', pad=8)

    fig.suptitle(f'Pixel Sensitivity — "{predicted_class.title()}" prediction',
                 color='white', fontsize=12, fontweight='700', y=1.02)
    fig.tight_layout(pad=1.5)
    return _b64_fig(fig)


def _plot_proba_radar(proba_dict):
    """Radar / spider chart of class probabilities."""
    labels = list(proba_dict.keys())
    vals   = [proba_dict[l] / 100 for l in labels]
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals   += vals[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor('#0d0d1f')
    ax.spines['polar'].set_color('#2a2a55')
    ax.tick_params(colors='#6272a4', labelsize=9)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.title() for l in labels], color='#a0a8c8', fontsize=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.grid(color='#1e1e40', linewidth=0.6)

    ax.plot(angles, vals, 'o-', linewidth=2, color='#6c63ff')
    ax.fill(angles, vals, alpha=0.25, color='#6c63ff')

    # Highlight predicted class dot
    max_i  = vals.index(max(vals[:-1]))
    ax.plot(angles[max_i], vals[max_i], 'o', markersize=12,
            color='#ffd93d', zorder=5)

    ax.set_title('Class Probability Radar', color='white',
                 fontsize=11, fontweight='700', pad=18)
    fig.tight_layout()
    return _b64_fig(fig)


# ── Cache + main predict ──────────────────────────────────────────────────────
_CNN_CACHE = {}   # key → (model, scaler, train_acc)


def predict_cnn(file_bytes: bytes, params=None):
    p        = params or {}
    h1       = int(p.get('h1', 128))
    h2       = int(p.get('h2', 64))
    cache_key = f"cnn_{h1}_{h2}"

    x_raw, thumb_b64 = preprocess_image(file_bytes)

    if cache_key in _CNN_CACHE:
        model, scaler, train_acc = _CNN_CACHE[cache_key]
    else:
        X_train, y_train = _load_dataset()
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_train)
        model  = MLPClassifier(
            hidden_layer_sizes=(h1, h2),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=80,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            verbose=False,
        )
        model.fit(X_sc, y_train)
        train_acc = round(accuracy_score(y_train, model.predict(X_sc)) * 100, 1)
        _CNN_CACHE[cache_key] = (model, scaler, train_acc)

    x_sc  = scaler.transform([x_raw])
    pred  = model.predict(x_sc)[0]
    proba = model.predict_proba(x_sc)[0]
    predicted_class = CLASSES[pred]
    proba_dict = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(CLASSES, proba)
    }
    confidence = round(float(proba[pred]) * 100, 1)

    return {
        'skin_type':    predicted_class,
        'confidence':   confidence,
        'probabilities': proba_dict,
        'train_acc':    train_acc,
        'n_iter':       int(model.n_iter_),
        'image_b64':    thumb_b64,
        'plot_sensitivity': _plot_sensitivity(model, scaler, x_raw, predicted_class, file_bytes),
        'plot_radar':       _plot_proba_radar(proba_dict),
    }
