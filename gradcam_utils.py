# === Multi-layer Grad-CAM + Grad-CAM++ + Face-mesh analysis (DISPLAY INLINE, temporal stats + drift + per-region drift + heatmap VIDEO, show video inline) ===
# Paste this entire cell. Requires ModelArcAux & build_efficientnetv2m_backbone defined earlier.

from typing import Callable, List, Tuple, Dict, Any, Optional
import os, json, math, time
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib.pyplot as plt

# imageio left for compatibility if you want it elsewhere (not used for video writing here)
try:
    import imageio.v2 as imageio
except Exception:
    import imageio

# For inline display in notebooks (optional for Streamlit compatibility)
try:
    from IPython.display import Image as IPyImage, display, HTML, Video as IPyVideo
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    # Define dummy functions for Streamlit compatibility
    def display(*args, **kwargs):
        pass
    def IPyVideo(*args, **kwargs):
        return None

# Optional: mediapipe face mesh
try:
    import mediapipe as mp
    MP_FACE_MESH_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    MP_FACE_MESH_AVAILABLE = False

# Preprocess
try:
    WEIGHTS = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    DEFAULT_PREPROCESS = WEIGHTS.transforms()
except Exception:
    DEFAULT_PREPROCESS = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

# ---------- helpers (same as before) ----------
def find_conv_layers(module: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    convs = []
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            convs.append((name, m))
    return convs

class GradCAMMulti:
    def __init__(self, model: torch.nn.Module, target_layers: List[Tuple[str, torch.nn.Module]]):
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def make_forward(name):
            def forward_hook(module, inp, out):
                self.activations[name] = out.detach()
            return forward_hook

        def make_backward(name):
            def backward_hook(module, grad_in, grad_out):
                self.gradients[name] = grad_out[0].detach()
            return backward_hook

        for name, layer in self.target_layers:
            self.hooks.append(layer.register_forward_hook(make_forward(name)))
            # register_backward_hook used for compatibility
            self.hooks.append(layer.register_backward_hook(make_backward(name)))

    def remove_hooks(self):
        for h in self.hooks:
            try: h.remove()
            except: pass
        self.hooks = []

    def _compute_cam_from_activation_grad(self, activation: torch.Tensor, gradient: torch.Tensor, method: str = 'gradcam') -> np.ndarray:
        act = activation.cpu()
        grad = gradient.cpu()
        if method == 'gradcam':
            weights = grad.mean(dim=(1,2))
            cam = (weights.view(-1,1,1) * act).sum(dim=0).numpy()
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() + 1e-8)
            else:
                cam = np.zeros_like(cam)
            return cam.astype(np.float32)
        elif method == 'gradcam++':
            grads = grad
            activ = act
            grads2 = grads ** 2
            grads3 = grads ** 3
            eps = 1e-8
            sum_grad2 = grads2.sum(dim=(1,2))
            sum_activ_grad3 = (activ * grads3).sum(dim=(1,2))
            denom = (2.0 * sum_grad2 + sum_activ_grad3).view(-1,1,1)
            alpha = grads2 / (denom + eps)
            positive_grad = torch.relu(grads)
            weights = (alpha * positive_grad).sum(dim=(1,2))
            cam = (weights.view(-1,1,1) * activ).sum(dim=0).numpy()
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() + 1e-8)
            else:
                cam = np.zeros_like(cam)
            return cam.astype(np.float32)
        else:
            raise ValueError("Unknown method")

    def compute(self, input_tensor: torch.Tensor, target_scalar: torch.Tensor, method: str='gradcam') -> Dict[str, np.ndarray]:
        self.activations = {}
        self.gradients = {}
        self.model.zero_grad()
        try:
            outputs = self.model(input_tensor, None)
            logits_aux = outputs[1]
        except TypeError:
            outputs = self.model(input_tensor)
            logits_aux = outputs[1]
        scalar = target_scalar.view(-1)[0]
        scalar.backward(retain_graph=True)
        cams = {}
        for name, _ in self.target_layers:
            act = self.activations.get(name, None)
            grad = self.gradients.get(name, None)
            if act is None or grad is None:
                continue
            act0 = act[0]; grad0 = grad[0]
            try:
                cam = self._compute_cam_from_activation_grad(act0, grad0, method=method)
            except Exception:
                cam = self._compute_cam_from_activation_grad(act0, grad0, method='gradcam')
            cams[name] = cam
        return cams

def build_face_region_masks_from_mediapipe(face_rgb: np.ndarray, mesh_landmarks) -> Dict[str, np.ndarray]:
    h, w = face_rgb.shape[:2]
    pts = []
    for lm in mesh_landmarks:
        x = min(max(int(lm.x * w), 0), w-1)
        y = min(max(int(lm.y * h), 0), h-1)
        pts.append((x,y))
    LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    MOUTH_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    NOSE_IDX = [1, 2, 98, 327, 168]
    LEFT_CHEEK_IDX = [50, 101, 36, 112, 189]
    RIGHT_CHEEK_IDX = [280, 349, 263, 361, 418]
    FOREHEAD_IDX = [10, 338, 297, 332, 284]
    regions = {
        "left_eye": LEFT_EYE_IDX, "right_eye": RIGHT_EYE_IDX, "mouth": MOUTH_IDX,
        "nose": NOSE_IDX, "left_cheek": LEFT_CHEEK_IDX, "right_cheek": RIGHT_CHEEK_IDX, "forehead": FOREHEAD_IDX
    }
    masks = {}
    for name, idxs in regions.items():
        poly = [pts[i] for i in idxs if i < len(pts)]
        if len(poly) < 3:
            masks[name] = np.zeros((h,w), dtype=bool)
            continue
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(poly, dtype=np.int32), 1)
        masks[name] = mask.astype(bool)
    return masks

def build_face_region_masks_from_bbox(face_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = face_rgb.shape[:2]
    mouth = np.zeros((h,w), dtype=bool); y1=int(h*0.65); mouth[y1:h, int(w*0.2):int(w*0.8)] = True
    eye_left = np.zeros((h,w), dtype=bool); eye_right = np.zeros((h,w), dtype=bool)
    ey_y1,ey_y2 = int(h*0.15), int(h*0.45)
    eye_left[ey_y1:ey_y2, int(w*0.05):int(w*0.48)] = True
    eye_right[ey_y1:ey_y2, int(w*0.52):int(w*0.95)] = True
    nose = np.zeros((h,w), dtype=bool); nose[int(h*0.35):int(h*0.65), int(w*0.38):int(w*0.62)] = True
    left_cheek = np.zeros((h,w), dtype=bool); left_cheek[int(h*0.35):int(h*0.7), int(w*0.05):int(w*0.35)] = True
    right_cheek = np.zeros((h,w), dtype=bool); right_cheek[int(h*0.35):int(h*0.7), int(w*0.65):int(w*0.95)] = True
    forehead = np.zeros((h,w), dtype=bool); forehead[0:int(h*0.15), int(w*0.15):int(w*0.85)] = True
    return {"left_eye": eye_left, "right_eye": eye_right, "mouth": mouth, "nose": nose,
            "left_cheek": left_cheek, "right_cheek": right_cheek, "forehead": forehead}

def compute_region_stats_from_cam(cam_resized: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    stats = {}
    for name, mask in masks.items():
        if mask.sum() == 0:
            stats[name] = 0.0
        else:
            stats[name] = float(cam_resized[mask].mean())
    return stats

def make_overlay_array(face_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    orig_np = np.array(face_pil)  # RGB
    h, w = orig_np.shape[:2]
    cam_resized = cv2.resize((cam * 255).astype(np.uint8), (w, h))
    heat_bgr = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = (heat_rgb.astype(float) * alpha + orig_np.astype(float) * (1 - alpha)).astype(np.uint8)
    return orig_np, heat_rgb, overlay

def make_overlay_array(face_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    orig_np = np.array(face_pil)  # RGB
    h, w = orig_np.shape[:2]
    cam_resized = cv2.resize((cam * 255).astype(np.uint8), (w, h))
    heat_bgr = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = (heat_rgb.astype(float) * alpha + orig_np.astype(float) * (1 - alpha)).astype(np.uint8)
    return orig_np, heat_rgb, overlay


# Model Architecture Classes
class ModelArcAux(nn.Module):
    """Model architecture with auxiliary output for deepfake detection"""
    
    def __init__(self, backbone_builder: Callable, num_classes: int = 2, 
                 embedding_dim: int = 512, dropout_rate: float = 0.5, pretrained: bool = False):
        super().__init__()
        # Build backbone with embedding_dim as num_classes to match checkpoint
        self.backbone = backbone_builder(num_classes=embedding_dim, dropout_rate=dropout_rate)
        
        # Arc face classifier (main output) - takes embeddings and outputs num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, num_classes)
        )
        
        # Auxiliary classifier - takes embeddings and outputs 1 for binary classification
        self.aux_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, 1)
        )
        
    def forward(self, x, labels=None):
        # Get embeddings from backbone
        embeddings = self.backbone(x)
        
        # Main classifier output
        arc_logits = self.classifier(embeddings)
        
        # Auxiliary classifier output  
        aux_logits = self.aux_classifier(embeddings)
        
        return arc_logits, aux_logits, embeddings


def build_efficientnetv2m_backbone(num_classes=1, dropout_rate=0.5):
    """Build EfficientNetV2-M backbone"""
    try:
        from torchvision.models import efficientnet_v2_m
        model = efficientnet_v2_m(weights=None)
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        return model
    except ImportError:
        # Fallback to a simple CNN if EfficientNet not available
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )


# Minimal checkpoint loader helper
def init_model_from_checkpoint(checkpoint_path: str, device: str, backbone_builder: Callable,
                               embedding_dim: int = 512, dropout_rate: float = 0.5, map_location: str = "cpu"):
    model = ModelArcAux(backbone_builder=backbone_builder, num_classes=2,
                        embedding_dim=embedding_dim, dropout_rate=dropout_rate, pretrained=False)
    
    try:
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(state, dict) and len(state) > 0 and list(state.keys())[0].startswith('module.'):
            state = {k[7:]: v for k,v in state.items()}
        if isinstance(state, dict) and 'state_dict' in state:
            sd = state['state_dict']
            if isinstance(sd, dict) and len(sd) > 0 and list(sd.keys())[0].startswith('module.'):
                sd = {k[7:]: v for k,v in sd.items()}
            state = sd
        
        # Load state dict with more flexible handling
        model_dict = model.state_dict()
        pretrained_dict = {}
        
        for k, v in state.items():
            if k in model_dict:
                # Check if shapes match
                if model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                else:
                    print(f"Skipping layer {k} due to shape mismatch: {model_dict[k].shape} vs {v.shape}")
            else:
                print(f"Skipping unknown layer: {k}")
        
        # Update model dict and load
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print(f"Successfully loaded {len(pretrained_dict)} layers from checkpoint")
        
    except Exception as e:
        print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
        print("Using randomly initialized model")
    
    model.to(device)
    model.eval()
    return model

# ---------- MAIN: with temporal stability & attention drift & per-region drift & heatmap VIDEO ----------
def run_multilayer_gradcam_display(
    checkpoint_path: str,
    video_path: str,
    backbone_builder: Callable,
    device: str = "cpu",
    num_sampled_frames: int = 8,
    preprocess = DEFAULT_PREPROCESS,
    threshold: float = 0.5,
    method: str = "gradcam++",
    layers_to_use: Optional[List[str]] = None,
    use_mediapipe: bool = True,
    max_display: int = 4,
    heatmap_fps: int = 6,
    heatmap_name: str = "temporal_heatmap"
) -> Dict[str, Any]:
    """
    Runs multilayer Grad-CAM/Grad-CAM++, displays inline, computes per-region drift and
    creates a temporal heatmap MP4 (averaging cams across layers). Returns a rich report dict.
    The MP4 will be displayed inline in a Jupyter notebook when produced.
    """
    report = {"video": video_path, "frames": [], "video_level_score": None, "final_prediction": None, "method": method}

    model = init_model_from_checkpoint(checkpoint_path, device=device, backbone_builder=backbone_builder,
                                       embedding_dim=512, dropout_rate=0.5, map_location=device)
    model_for_search = model.module if isinstance(model, torch.nn.DataParallel) else model
    backbone = model_for_search.backbone

    convs = find_conv_layers(backbone)
    if not convs:
        raise RuntimeError("No conv layers found.")
    if layers_to_use:
        selected = []
        for name, layer in convs:
            for s in layers_to_use:
                if s in name:
                    selected.append((name, layer))
                    break
        if not selected:
            selected = convs[-3:]
    else:
        selected = convs[-3:]
    layer_names = [n for n,_ in selected]
    print("Using layers:", layer_names)

    grad_multi = GradCAMMulti(model_for_search, selected)

    mp_face = None
    if use_mediapipe and MP_FACE_MESH_AVAILABLE:
        mp_face = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)
        print("Mediapipe FaceMesh enabled.")
    else:
        if use_mediapipe:
            print("Mediapipe not available — using bbox-based region masks.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0:
        raise RuntimeError("Video has zero frames.")
    frame_indices = list(np.linspace(0, max(0,total_frames-1), num_sampled_frames, dtype=int))

    # storage for region timeseries and centroids
    per_layer_region_ts: Dict[str, Dict[str, List[float]]] = {ln: {} for ln in layer_names}
    per_layer_centroids: Dict[str, List[Tuple[float,float]]] = {ln: [] for ln in layer_names}
    display_rows = []
    frame_scores = []

    # For heatmap video: collect combined overlay frames (RGB arrays)
    heatmap_frames = []
    face_sizes = []  # store sizes for writer

    for idx in tqdm(frame_indices, desc="Processing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # detect face bbox
        try:
            import dlib
            detector = dlib.get_frontal_face_detector()
            dets = detector(gray, 0)
            if len(dets) == 0:
                x1,y1,x2,y2 = 0,0,frame_bgr.shape[1], frame_bgr.shape[0]
            else:
                d = dets[0]
                x1,y1,x2,y2 = max(0,d.left()), max(0,d.top()), min(frame_bgr.shape[1],d.right()), min(frame_bgr.shape[0],d.bottom())
        except Exception:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(haar_xml)
            dets = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(dets) == 0:
                x1,y1,x2,y2 = 0,0,frame_bgr.shape[1], frame_bgr.shape[0]
            else:
                x,y,w,h = dets[0]; x1,y1,x2,y2 = x,y,x+w,y+h

        face_crop_bgr = frame_bgr[y1:y2, x1:x2].copy()
        if face_crop_bgr.size == 0:
            face_crop_bgr = frame_bgr.copy()
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)
        face_sizes.append((face_rgb.shape[1], face_rgb.shape[0]))  # (w,h)

        # forward for prob
        tensor = preprocess(pil_face).unsqueeze(0).to(device)
        model_for_search.eval()
        with torch.no_grad():
            try:
                logits_arc, logits_aux, _ = model_for_search(tensor, None)
            except TypeError:
                logits_arc, logits_aux, _ = model_for_search(tensor)
        prob = float(torch.sigmoid(logits_aux).item())
        frame_scores.append(prob)

        # compute multilayer cams
        model_for_search.zero_grad()
        try:
            logits_arc2, logits_aux2, _ = model_for_search(tensor, None)
        except TypeError:
            logits_arc2, logits_aux2, _ = model_for_search(tensor)
        cams = grad_multi.compute(tensor, logits_aux2, method=method)  # dict name->cam (Hf x Wf 0..1)

        # build combined cam as average across selected layers (aligned to face size)
        cam_list_resized = []
        for lname, cam in cams.items():
            cam_resized_face = cv2.resize((cam * 255).astype(np.uint8), (face_rgb.shape[1], face_rgb.shape[0])).astype(np.float32) / 255.0
            cam_list_resized.append(cam_resized_face)
        if len(cam_list_resized) == 0:
            combined_cam_face = np.zeros((face_rgb.shape[0], face_rgb.shape[1]), dtype=np.float32)
        else:
            combined_cam_face = np.mean(np.stack(cam_list_resized, axis=0), axis=0)

        # Save overlay frame for heatmap video (RGB)
        _, heat_rgb_comb, overlay_comb = make_overlay_array(pil_face, combined_cam_face, alpha=0.45)
        heatmap_frames.append(overlay_comb)  # RGB uint8

        per_layer_display = []
        for lname in layer_names:
            cam = cams.get(lname, None)
            if cam is None:
                cam = np.zeros((1,1), dtype=np.float32)
                cam_resized = np.zeros((face_rgb.shape[0], face_rgb.shape[1]), dtype=np.float32)
            else:
                cam_resized = cv2.resize((cam * 255).astype(np.uint8), (face_rgb.shape[1], face_rgb.shape[0])).astype(np.float32) / 255.0

            # compute centroid (attention center) normalized [0,1] coords
            h_face, w_face = cam_resized.shape[:2]
            total_mass = cam_resized.sum() + 1e-8
            ys, xs = np.mgrid[0:h_face, 0:w_face]
            cx = (cam_resized * xs).sum() / total_mass
            cy = (cam_resized * ys).sum() / total_mass
            cx_n = cx / float(max(1, w_face-1))
            cy_n = cy / float(max(1, h_face-1))
            per_layer_centroids[lname].append((cx_n, cy_n))

            # face-mesh masks
            if mp_face is not None:
                mp_img = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                results = mp_face.process(cv2.cvtColor(mp_img, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                    mesh = results.multi_face_landmarks[0].landmark
                    masks = build_face_region_masks_from_mediapipe(face_rgb, mesh)
                else:
                    masks = build_face_region_masks_from_bbox(face_rgb)
            else:
                masks = build_face_region_masks_from_bbox(face_rgb)

            region_stats = compute_region_stats_from_cam(cam_resized, masks)

            # append region timeseries values (per-layer)
            if lname not in per_layer_region_ts:
                per_layer_region_ts[lname] = {k: [] for k in masks.keys()}
            for k in masks.keys():
                per_layer_region_ts[lname].setdefault(k, []).append(region_stats.get(k, 0.0))

            orig_np, heat_np, overlay_np = make_overlay_array(pil_face, cam_resized, alpha=0.45)
            per_layer_display.append({
                "layer": lname,
                "cam_resized": cam_resized,
                "orig_np": orig_np,
                "heat_np": heat_np,
                "overlay_np": overlay_np,
                "region_stats": region_stats,
                "centroid_norm": (cx_n, cy_n)
            })

        display_rows.append({
            "frame_index": int(idx),
            "prob": prob,
            "per_layer": per_layer_display,
            "bbox": [int(x1),int(y1),int(x2),int(y2)],
            "combined_cam": combined_cam_face
        })

    cap.release()
    grad_multi.remove_hooks()

    # video-level
    if len(frame_scores):
        video_score = float(np.mean(frame_scores))
        report["video_level_score"] = video_score
        report["final_prediction"] = "Fake" if video_score > threshold else "Real"
    else:
        report["video_level_score"] = 0.0
        report["final_prediction"] = "N/A"

    # ---------- Temporal stability per layer & overall ----------
    overall_region_means = {}
    per_layer_region_summary = {}
    eps = 1e-8
    for lname, ts_dict in per_layer_region_ts.items():
        per_layer_region_summary[lname] = {}
        for region, vals in ts_dict.items():
            arr = np.array(vals, dtype=np.float32)
            mean = float(arr.mean()) if arr.size>0 else 0.0
            std = float(arr.std()) if arr.size>0 else 0.0
            cov = std / (mean + eps)
            stability = max(0.0, min(1.0, 1.0 - cov))
            per_layer_region_summary[lname][region] = {"mean": mean, "std": std, "stability": stability}
            overall_region_means[region] = overall_region_means.get(region, 0.0) + mean

    for region in overall_region_means:
        overall_region_means[region] = overall_region_means[region] / max(1, len(per_layer_region_ts.keys()))
    top_region = max(overall_region_means.items(), key=lambda x: x[1])[0] if overall_region_means else None

    # ---------- Attention drift: compute centroid path length per layer and combined ----------
    per_layer_drift = {}
    combined_centroid_path = []
    n_frames_effective = max(1, len(display_rows))
    for i_frame in range(n_frames_effective):
        pts = []
        for lname in layer_names:
            pts.append(per_layer_centroids.get(lname, [(0.5,0.5)])[i_frame] if i_frame < len(per_layer_centroids.get(lname, [])) else (0.5,0.5))
        avg_x = float(np.mean([p[0] for p in pts]))
        avg_y = float(np.mean([p[1] for p in pts]))
        combined_centroid_path.append((avg_x, avg_y))

    for lname in layer_names:
        pts = per_layer_centroids.get(lname, [])
        if len(pts) < 2:
            per_layer_drift[lname] = {"path_length": 0.0, "normalized": 0.0}
            continue
        dist = 0.0
        for a,b in zip(pts[:-1], pts[1:]):
            dist += math.hypot(a[0]-b[0], a[1]-b[1])
        per_layer_drift[lname] = {"path_length": float(dist), "normalized": float(dist / math.sqrt(2))}
    if len(combined_centroid_path) < 2:
        combined_drift = 0.0
    else:
        cd = 0.0
        for a,b in zip(combined_centroid_path[:-1], combined_centroid_path[1:]):
            cd += math.hypot(a[0]-b[0], a[1]-b[1])
        combined_drift = float(cd)

    # ---------- Per-region centroid drift (left_eye, right_eye, mouth, nose) ----------
    region_names = ["left_eye", "right_eye", "mouth", "nose"]
    per_region_centroid_ts: Dict[str, List[Tuple[float,float]]] = {r: [] for r in region_names}
    for row in display_rows:
        combined_cam = row["combined_cam"]
        face_rgb = row["per_layer"][0]["orig_np"]
        if mp_face is not None:
            mp_img = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            results = mp_face.process(cv2.cvtColor(mp_img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                mesh = results.multi_face_landmarks[0].landmark
                masks_all = build_face_region_masks_from_mediapipe(face_rgb, mesh)
            else:
                masks_all = build_face_region_masks_from_bbox(face_rgb)
        else:
            masks_all = build_face_region_masks_from_bbox(face_rgb)
        for r in region_names:
            mask = masks_all.get(r, np.zeros_like(combined_cam, dtype=bool))
            if mask.sum() == 0:
                per_region_centroid_ts[r].append((0.5,0.5))
                continue
            ys, xs = np.mgrid[0:combined_cam.shape[0], 0:combined_cam.shape[1]]
            mass = combined_cam * mask
            total_mass = mass.sum() + 1e-8
            cx = (mass * xs).sum() / total_mass
            cy = (mass * ys).sum() / total_mass
            per_region_centroid_ts[r].append((float(cx / max(1, combined_cam.shape[1]-1)), float(cy / max(1, combined_cam.shape[0]-1))))

    per_region_drift = {}
    for r, pts in per_region_centroid_ts.items():
        if len(pts) < 2:
            per_region_drift[r] = 0.0
            continue
        dist = 0.0
        for a,b in zip(pts[:-1], pts[1:]):
            dist += math.hypot(a[0]-b[0], a[1]-b[1])
        per_region_drift[r] = float(dist)

    # ---------- Create temporal heatmap MP4 ----------
    heatmap_video_path = None
    if len(heatmap_frames) > 0:
        max_w = max([w for w,h in face_sizes])
        max_h = max([h for w,h in face_sizes])
        standardized_frames = []
        for i, f_rgb in enumerate(heatmap_frames):
            h_f, w_f = f_rgb.shape[:2]
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8) + 0
            x0 = (max_w - w_f) // 2
            y0 = (max_h - h_f) // 2
            canvas[y0:y0+h_f, x0:x0+w_f] = f_rgb
            standardized_frames.append(canvas)
        # write MP4 using OpenCV
        heatmap_video_path = os.path.join(os.getcwd(), heatmap_name + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(heatmap_video_path, fourcc, float(heatmap_fps), (max_w, max_h))
        for fr in standardized_frames:
            # convert RGB->BGR for OpenCV
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()

        # DISPLAY THE MP4 INLINE IMMEDIATELY (not just print path)
        try:
            display(IPyVideo(heatmap_video_path, embed=True))
        except Exception:
            try:
                display(HTML(f'<video controls src="{heatmap_video_path}"></video>'))
            except Exception:
                print("Video created at:", heatmap_video_path)

    # Attach analysis to report
    report["frames"] = display_rows  # Add this crucial line!
    report["layers"] = layer_names
    report["per_layer_region_summary"] = per_layer_region_summary
    report["overall_region_means"] = overall_region_means
    report["top_region_overall"] = top_region
    report["per_layer_drift"] = per_layer_drift
    report["combined_attention_drift"] = combined_drift
    report["per_region_drift"] = per_region_drift
    report["num_sampled_frames"] = len(display_rows)
    report["heatmap_video_path"] = heatmap_video_path

    # ---------------- Display main grid (top frames) ----------------
    top_rows = sorted(display_rows, key=lambda x: x['prob'], reverse=True)[:max_display]
    n_show = len(top_rows)
    if n_show == 0:
        print("No frames to display.")
        return report

    num_cols = 1 + len(layer_names)
    fig_w = 3 * num_cols
    fig_h = 3 * n_show
    plt.figure(figsize=(fig_w, fig_h))
    for r_i, row in enumerate(top_rows):
        for c_i in range(num_cols):
            ax = plt.subplot(n_show, num_cols, r_i * num_cols + c_i + 1)
            ax.axis('off')
            if c_i == 0:
                img = row['per_layer'][0]['orig_np']
                ax.imshow(img)
                ax.set_title(f"Frame {row['frame_index']} (p={row['prob']:.2f})\norig")
            else:
                layer_entry = row['per_layer'][c_i - 1]
                ax.imshow(layer_entry['overlay_np'])
                stats = layer_entry['region_stats']
                stat_label = ", ".join([f"{k[:4]}:{v:.2f}" for k,v in stats.items() if v>0][:4])
                ax.set_title(f"{layer_entry['layer']}\n{stat_label}")
    plt.suptitle(f"Top {n_show} frames — multilayer {method} overlays (video score {report['video_level_score']:.3f} -> {report['final_prediction']})", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

    # ---------- Show best frame with per-layer thumbnails ----------
    best = top_rows[0]
    n_layers = len(layer_names)
    plt.figure(figsize=(3*(n_layers+1), 4))
    plt.subplot(1, n_layers+1, 1); plt.imshow(best['per_layer'][0]['orig_np']); plt.axis('off'); plt.title(f"Best Frame {best['frame_index']} orig (p={best['prob']:.3f})")
    for i, layer_entry in enumerate(best['per_layer']):
        plt.subplot(1, n_layers+1, i+2); plt.imshow(layer_entry['overlay_np']); plt.axis('off'); plt.title(layer_entry['layer'])
    plt.suptitle("Best frame: original + per-layer overlays", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

    # ---------- Temporal region plots (averaged across layers) ----------
    regions = list(next(iter(per_layer_region_ts.values())).keys()) if per_layer_region_ts else []
    combined_region_ts = {}
    for r in regions:
        combined_region_ts[r] = []
        for i in range(n_frames_effective):
            vals = []
            for lname in layer_names:
                arr = per_layer_region_ts.get(lname, {}).get(r, [])
                if i < len(arr):
                    vals.append(arr[i])
            combined_region_ts[r].append(float(np.mean(vals)) if len(vals)>0 else 0.0)
    plt.figure(figsize=(10,4 + 0.3*len(regions)))
    for r in regions:
        plt.plot(range(len(combined_region_ts[r])), combined_region_ts[r], label=r)
    plt.xlabel("sampled frame index (temporal order)")
    plt.ylabel("mean CAM intensity")
    plt.title("Region CAM intensity over sampled frames (averaged across layers)")
    plt.legend(bbox_to_anchor=(1.01, 1.0))
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # ---------- Per-region centroid drift visualization (bar) ----------
    plt.figure(figsize=(8,4))
    names = list(per_region_drift.keys())
    values = [per_region_drift[n] for n in names]
    plt.bar(names, values)
    plt.title("Per-region centroid path length (drift across sampled frames)")
    plt.ylabel("path length (normalized coords)")
    plt.tight_layout()
    plt.show()

    # ---------- Per-region centroid path plots (each region separately) ----------
    for r in region_names:
        pts = per_region_centroid_ts.get(r, [])
        if len(pts) == 0:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.figure(figsize=(6,4))
        plt.plot(range(len(xs)), xs, label=f"{r} x (norm)")
        plt.plot(range(len(ys)), ys, label=f"{r} y (norm)")
        plt.xlabel("sampled frame index")
        plt.ylabel("normalized centroid coord")
        plt.title(f"Centroid coordinates over time — region: {r} (drift={per_region_drift.get(r,0.0):.3f})")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

        # also plot 2D path on best face
        best_face_np = best['per_layer'][0]['orig_np']
        h_b, w_b = best_face_np.shape[:2]
        pts_px = [(int(x*w_b), int(y*h_b)) for (x,y) in pts]
        plt.figure(figsize=(4,4))
        plt.imshow(best_face_np)
        xs_px = [p[0] for p in pts_px]; ys_px = [p[1] for p in pts_px]
        plt.plot(xs_px, ys_px, '-o', color='cyan', markersize=6)
        for i,(xx,yy) in enumerate(pts_px):
            plt.text(xx+2, yy+2, str(i), color='yellow', fontsize=8)
        plt.title(f"{r} centroid path on best face")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ---------- Attention drift visualization (combined centroid path) ----------
    best_face_np = best['per_layer'][0]['orig_np']
    h_b, w_b = best_face_np.shape[:2]
    pts_px = [(int(x*w_b), int(y*h_b)) for (x,y) in combined_centroid_path]
    plt.figure(figsize=(6,6))
    plt.imshow(best_face_np)
    xs = [p[0] for p in pts_px]; ys = [p[1] for p in pts_px]
    plt.scatter(xs, ys, c='yellow', s=60, edgecolors='black')
    for i in range(len(pts_px)-1):
        x0,y0 = pts_px[i]; x1,y1 = pts_px[i+1]
        plt.arrow(x0, y0, x1-x0, y1-y0, color='cyan', head_width=4, length_includes_head=True, alpha=0.8)
    plt.title(f"Combined centroid path (attention drift = {combined_drift:.3f})")
    plt.axis('off')
    plt.show()

    # ---------- Print stability summary (overall) ----------
    print("\nRegion importance (avg across layers) and temporal stability (1 - cov):")
    for reg, meanv in sorted(overall_region_means.items(), key=lambda x: x[1], reverse=True):
        stabilities = []
        for lname in per_layer_region_summary.keys():
            sdict = per_layer_region_summary[lname].get(reg, None)
            if sdict:
                stabilities.append(sdict["stability"])
        stability_avg = float(np.mean(stabilities)) if stabilities else 0.0
        print(f"  {reg:12s} mean={meanv:.3f}  stability={stability_avg:.3f}")
    if top_region:
        print(f"\nMost important region overall: {top_region}")

    # ---------- Notify video location ----------
    if report["heatmap_video_path"] is None and heatmap_video_path is not None:
        report["heatmap_video_path"] = heatmap_video_path
    if report.get("heatmap_video_path"):
        print(f"\nTemporal heatmap video created at: {report['heatmap_video_path']}")
        print("In Streamlit use: st.video('temporal_heatmap.mp4') to display it.")

    return report

# ---------------- example usage ----------------
if __name__ == "__main__":
    CHECKPOINT = "/kaggle/input/newestmodel2/best_staged_model (1).pth"
    VIDEO = "/kaggle/input/celeb-df-v2/YouTube-real/00022.mp4"
    report = run_multilayer_gradcam_display(
        checkpoint_path=CHECKPOINT,
        video_path=VIDEO,
        backbone_builder=build_efficientnetv2m_backbone,
        device="cpu",
        num_sampled_frames=12,
        preprocess=DEFAULT_PREPROCESS,
        threshold=0.84,
        method="gradcam++",
        layers_to_use=['features.7.3.block.3.0', 'features.7.4.block.3.0', 'features.8.0'],
        use_mediapipe=True,
        max_display=4,
        heatmap_fps=6,
        heatmap_name="temporal_heatmap"
    )
    # the function already displays the MP4 inline when produced; still print the report path and dict
    if report.get("heatmap_video_path"):
        try:
            display(IPyVideo(report["heatmap_video_path"], embed=True))
        except Exception:
            print("Video available at:", report["heatmap_video_path"])
    print(json.dumps(report, indent=2))