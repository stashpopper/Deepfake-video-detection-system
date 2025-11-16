# app.py
import streamlit as st
import tempfile
import os
import io
import json
import math
from typing import Callable, List, Tuple, Dict, Any, Optional
from pathlib import Path
import inspect
import time

# standard ML / CV libs
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from tqdm import tqdm
import pandas as pd

# optional libs that may not be installed; functions handle absence
try:
    import mediapipe as mp
    MP_FACE_MESH_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    MP_FACE_MESH_AVAILABLE = False

# ---------------------------
# GLOBAL CONFIG
# ---------------------------
MODEL_PATH = r"C:\Users\HP\OneDrive\Documents\vitpretrain\deepfakedetection\Deepfake-video-detection-system\best_staged_model.pth"  # <-- set your model path here or via UI
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# ---------------------------
# ---------------------------
# === START: Model & Training code (from your second script) ===
# This block preserves the model/training functionality you provided earlier.
# ---------------------------
# ---------------------------

import random
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

def build_efficientnetv2m_backbone(embedding_dim=512, pretrained=True, dropout_rate=0.4):
    """Builds a pre-trained EfficientNetV2-M backbone."""
    try:
        weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
    except Exception:
        weights = None
    model = models.efficientnet_v2_m(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, embedding_dim),
    )
    return model

class SubCenterArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, K=3, s=64.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(out_features * K, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        normalized_feat = F.normalize(input)
        normalized_weight = F.normalize(self.weight)
        cosine = F.linear(normalized_feat, normalized_weight).view(-1, self.out_features, self.K)
        
        if label is None:
            logits, _ = torch.max(cosine, dim=2)
            return logits * self.s

        max_cosine_neg, _ = torch.max(cosine, dim=2)
        gt_cosine = torch.gather(cosine, 1, label.view(-1, 1, 1).expand(-1, -1, self.K)).squeeze(1)
        hard_positive_cos, _ = torch.max(gt_cosine, dim=1)
        hard_positive_cos = hard_positive_cos.clamp(-1, 1)

        sine = torch.sqrt(1.0 - torch.pow(hard_positive_cos, 2))
        phi = hard_positive_cos * self.cos_m - sine * self.sin_m
        phi = torch.where(hard_positive_cos > self.th, phi, hard_positive_cos - self.mm)

        one_hot = torch.zeros_like(max_cosine_neg).scatter_(1, label.view(-1, 1), 1.0)
        logits = (one_hot * phi.unsqueeze(1)) + ((1.0 - one_hot) * max_cosine_neg)
        return logits * self.s

class ModelArcAux(nn.Module):
    """A generic model class that accepts any backbone builder."""
    def __init__(self,
                 backbone_builder: Callable,
                 num_classes=2,
                 embedding_dim=512,
                 s=64.0, m=0.5, K=3,
                 pretrained=True,
                 dropout_rate=0.6):
        super().__init__()
        self.backbone = backbone_builder(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
        self.margin = SubCenterArcMarginProduct(embedding_dim, num_classes, s=s, m=m, K=K)
        self.aux_head = nn.Linear(embedding_dim, 1)

    def forward(self, x,labels=None):
        feats = self.backbone(x)
        logits_arc = self.margin(feats, labels)
        logits_aux = self.aux_head(feats).squeeze(1)
        return logits_arc, logits_aux, feats

# training/eval functions (kept intact)
def train_one_epoch(model, dataloader, ce_criterion, bce_criterion, optimizer, device, lambda_aux):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []
    progress_bar = tqdm(dataloader, desc="Train", leave=False)
    
    for i, (imgs, labels) in enumerate(progress_bar):
        if imgs is None or labels is None: continue
        imgs, labels_long, labels_float = imgs.to(device), labels.to(device).long(), labels.to(device).float()
        
        optimizer.zero_grad()
        logits_arc, logits_aux, _ = model(imgs, labels_long)
        loss = ce_criterion(logits_arc, labels_long) + lambda_aux * bce_criterion(logits_aux, labels_float)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        preds_all.extend(torch.sigmoid(logits_aux.detach()).cpu().numpy())
        labels_all.extend(labels_long.cpu().numpy())

    preds_all, labels_all = np.array(preds_all), np.array(labels_all)
    acc = accuracy_score(labels_all, preds_all > 0.5)
    auc = roc_auc_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all > 0.5)
    return running_loss / len(labels_all), acc, auc, f1

def eval_one_epoch(model, dataloader, ce_criterion, bce_criterion, device, lambda_aux):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []
    progress_bar = tqdm(dataloader, desc="Val", leave=False)

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(progress_bar):
            if imgs is None or labels is None: continue
            imgs, labels_long, labels_float = imgs.to(device), labels.to(device).long(), labels.to(device).float()
            
            logits_arc, logits_aux, _ = model(imgs, None)
            loss = ce_criterion(logits_arc, labels_long) + lambda_aux * bce_criterion(logits_aux, labels_float)
            
            running_loss += loss.item() * imgs.size(0)
            preds_all.extend(torch.sigmoid(logits_aux).cpu().numpy())
            labels_all.extend(labels_long.cpu().numpy())

    preds_all, labels_all = np.array(preds_all), np.array(labels_all)
    acc = accuracy_score(labels_all, preds_all > 0.5)
    auc = roc_auc_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all > 0.5)
    return running_loss / len(labels_all), acc, auc, f1



# ---------------------------
# === END: Model & Training code block ===
# ---------------------------

# ---------------------------
# ---------------------------
# === START: Grad-CAM & Prediction functions (from your first script) ===
# ---------------------------

# Preprocess default (kept same as before)
try:
    WEIGHTS = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    DEFAULT_PREPROCESS = WEIGHTS.transforms()
except Exception:
    DEFAULT_PREPROCESS = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

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

def init_model_from_checkpoint(checkpoint_path: str, device: str, backbone_builder: Callable,
                               embedding_dim: int = 512, dropout_rate: float = 0.5, map_location: str = "cpu"):
    model = ModelArcAux(backbone_builder=backbone_builder, num_classes=2,
                        embedding_dim=embedding_dim, dropout_rate=dropout_rate, pretrained=False)
    state = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(state, dict) and len(state) > 0 and list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k,v in state.items()}
    if isinstance(state, dict) and 'state_dict' in state:
        sd = state['state_dict']
        if isinstance(sd, dict) and len(sd) > 0 and list(sd.keys())[0].startswith('module.'):
            sd = {k[7:]: v for k,v in sd.items()}
        state = sd
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

# Prediction function (process_and_save_video_2) kept mostly intact
#import dlib
def process_and_save_video_2(model: nn.Module, mp_face_detector, video_path: str, output_path: str, device: str, threshold: float, num_frames_to_sample: int):
    """
    Processes a video in two passes on a specific number of sampled frames.
    Uses MediaPipe Face Detection (mp_face_detector) if provided; otherwise falls back to Haar cascade.
    """
    # --- Shared Setup ---
    image_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model.eval()

    parent_dir_name = os.path.basename(os.path.dirname(video_path)).lower()
    ground_truth_label = "Real" if "original" in parent_dir_name else "Fake"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0:
        st.error("Video has zero frames.")
        cap.release()
        return None

    # generate indices
    if num_frames_to_sample > 0 and num_frames_to_sample < total_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
    else:
        frame_indices = range(total_frames)

    # prepare Haar cascade in case mediapipe not available
    haar_cascade = None
    if not mp_face_detector:
        try:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            haar_cascade = cv2.CascadeClassifier(haar_xml)
        except Exception:
            haar_cascade = None

    # Pass 1: collect scores
    frame_scores = []
    sampled_info = []
    with torch.no_grad():
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            h_img, w_img = frame.shape[:2]
            face_bbox = None

            # Try MediaPipe detector first (if available)
            if mp_face_detector:
                try:
                    # MediaPipe expects RGB
                    results_fd = mp_face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results_fd and getattr(results_fd, "detections", None):
                        detection = results_fd.detections[0]
                        bboxC = detection.location_data.relative_bounding_box
                        if bboxC:
                            bx = bboxC.xmin; by = bboxC.ymin; bw = bboxC.width; bh = bboxC.height
                            x1 = int(max(0, bx * w_img)); y1 = int(max(0, by * h_img))
                            x2 = int(min(w_img, (bx + bw) * w_img)); y2 = int(min(h_img, (by + bh) * h_img))
                            face_bbox = (x1, y1, x2, y2)
                except Exception:
                    face_bbox = None

            # Fallback to Haar cascade if mediapipe failed or not available
            if face_bbox is None and haar_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    dets = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    if len(dets) != 0:
                        x,y,w,h = dets[0]
                        face_bbox = (int(x), int(y), int(x+w), int(y+h))
                except Exception:
                    face_bbox = None

            # If no face found, use full frame as fallback
            if face_bbox is None:
                x1, y1, x2, y2 = 0, 0, w_img, h_img
            else:
                x1, y1, x2, y2 = face_bbox

            face_crop_bgr = frame[y1:y2, x1:x2]
            if face_crop_bgr.size == 0:
                face_crop_bgr = frame.copy()
                x1, y1, x2, y2 = 0, 0, w_img, h_img

            face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_crop_rgb)
            image_tensor = image_transforms(pil_image).unsqueeze(0).to(device)

            if isinstance(model, nn.DataParallel):
                _, logits_aux, _ = model.module(image_tensor, labels=None)
            else:
                _, logits_aux, _ = model(image_tensor, labels=None)

            probability = float(torch.sigmoid(logits_aux).item())
            frame_scores.append(probability)
            sampled_info.append({"frame_idx": int(frame_idx), "prob": probability, "bbox": [int(x1), int(y1), int(x2), int(y2)]})

    cap.release()

    # video-level aggregation
    if not frame_scores:
        video_level_score = 0.0
        final_video_prediction = "N/A"
    else:
        video_level_score = float(np.mean(frame_scores))
        final_video_prediction = "Fake" if video_level_score > threshold else "Real"

    # Pass 2: write annotated video (only sampled frames written; others will be blank frames at same positions)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    final_pred_text = f"Final Prediction: {final_video_prediction}"
    gt_text = f"Actual Label: {ground_truth_label}"
    overlay_color = (0, 255, 0) if final_video_prediction == ground_truth_label else (0, 0, 255)

    with torch.no_grad():
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            h_img, w_img = frame.shape[:2]
            face_bbox = None

            if mp_face_detector:
                try:
                    results_fd = mp_face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results_fd and getattr(results_fd, "detections", None):
                        detection = results_fd.detections[0]
                        bboxC = detection.location_data.relative_bounding_box
                        if bboxC:
                            bx = bboxC.xmin; by = bboxC.ymin; bw = bboxC.width; bh = bboxC.height
                            x1 = int(max(0, bx * w_img)); y1 = int(max(0, by * h_img))
                            x2 = int(min(w_img, (bx + bw) * w_img)); y2 = int(min(h_img, (by + bh) * h_img))
                            face_bbox = (x1, y1, x2, y2)
                except Exception:
                    face_bbox = None

            if face_bbox is None and haar_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    dets = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    if len(dets) != 0:
                        x,y,w,h = dets[0]
                        face_bbox = (int(x), int(y), int(x+w), int(y+h))
                except Exception:
                    face_bbox = None

            if face_bbox is None:
                x1, y1, x2, y2 = 0, 0, w_img, h_img
            else:
                x1, y1, x2, y2 = face_bbox

            face_crop_bgr = frame[y1:y2, x1:x2]
            if face_crop_bgr.size > 0:
                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_crop_rgb)
                image_tensor = image_transforms(pil_image).unsqueeze(0).to(device)

                if isinstance(model, nn.DataParallel):
                    _, logits_aux, _ = model.module(image_tensor, labels=None)
                else:
                    _, logits_aux, _ = model(image_tensor, labels=None)
                prob = float(torch.sigmoid(logits_aux).item())
                label = "Fake" if prob > threshold else "Real"
                color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}: {prob:.2f}", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, final_pred_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, overlay_color, 2)
            cv2.putText(frame, gt_text, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            out.write(frame)

    cap.release()
    out.release()

    return {
        "video_level_score": video_level_score,
        "final_prediction": final_video_prediction,
        "frame_scores": sampled_info,
        "annotated_video": output_path
    }


# Grad-CAM display function (keeps original functionality, returns report dict)
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
    heatmap_name: str = "temporal_heatmap",
    target_size: int = 384,
    face_padding: float = 0.35
) -> Dict[str, Any]:
    """
    Streamlit-compatible version of run_multilayer_gradcam_display.
    Produces the same report dict but uses st.video/st.pyplot/st.image for display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = {"video": video_path, "frames": [], "video_level_score": None, "final_prediction": None, "method": method}

    # Preprocess to fixed size
    IMAGENET_MEAN = [0.485,0.456,0.406]
    IMAGENET_STD = [0.229,0.224,0.225]
    preprocess_resized = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

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
        print("Mediapipe FaceMesh enabled for landmarks.")
    else:
        if use_mediapipe:
            print("Mediapipe FaceMesh not available — using bbox-based region masks.")

    # MediaPipe Face Detection
    mp_face_detector = None
    MP_FACE_DET_AVAILABLE = False
    try:
        if use_mediapipe and 'mp' in globals() and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            mp_face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
            MP_FACE_DET_AVAILABLE = True
            print("Mediapipe FaceDetection enabled.")
    except Exception:
        MP_FACE_DET_AVAILABLE = False
        print("Mediapipe FaceDetection not available; will fall back to Haar cascade if needed.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0:
        raise RuntimeError("Video has zero frames.")
    frame_indices = list(np.linspace(0, max(0,total_frames-1), num_sampled_frames, dtype=int))

    per_layer_region_ts: Dict[str, Dict[str, List[float]]] = {ln: {} for ln in layer_names}
    per_layer_centroids: Dict[str, List[Tuple[float,float]]] = {ln: [] for ln in layer_names}
    display_rows = []
    frame_scores = []
    heatmap_frames = []
    face_sizes = []

    for idx in tqdm(frame_indices, desc="Processing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        h_img, w_img = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        x1,y1,x2,y2 = 0,0,w_img,h_img
        detected = False

        if MP_FACE_DET_AVAILABLE:
            try:
                results_fd = mp_face_detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                if results_fd and getattr(results_fd, "detections", None):
                    detection = results_fd.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    if bboxC:
                        bx = bboxC.xmin; by = bboxC.ymin; bw = bboxC.width; bh = bboxC.height
                        x1 = int(max(0, bx * w_img)); y1 = int(max(0, by * h_img))
                        x2 = int(min(w_img, (bx + bw) * w_img)); y2 = int(min(h_img, (by + bh) * h_img))
                        detected = True
            except Exception:
                detected = False

        if not detected:
            try:
                haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(haar_xml)
                dets = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(dets) != 0:
                    x,y,w,h = dets[0]; x1,y1,x2,y2 = x,y,x+w,y+h
                    detected = True
            except Exception:
                detected = False

        pad_horiz_frac = face_padding
        pad_top_frac = face_padding * 1.1
        pad_bottom_frac = face_padding * 0.6
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * pad_horiz_frac)
        pad_top = int(bh * pad_top_frac)
        pad_bottom = int(bh * pad_bottom_frac)
        x1_p = max(0, x1 - pad_x)
        x2_p = min(w_img, x2 + pad_x)
        y1_p = max(0, y1 - pad_top)
        y2_p = min(h_img, y2 + pad_bottom)

        if not detected:
            cx = w_img // 2; cy = h_img // 2
            half = int(min(w_img, h_img) * 0.5)
            x1_p = max(0, cx - half); x2_p = min(w_img, cx + half)
            y1_p = max(0, cy - half); y2_p = min(h_img, cy + half)

        if (x2_p - x1_p) < 20 or (y2_p - y1_p) < 20:
            x1_p, y1_p, x2_p, y2_p = 0, 0, w_img, h_img

        face_crop_bgr = frame_bgr[y1_p:y2_p, x1_p:x2_p].copy()
        if face_crop_bgr.size == 0:
            face_crop_bgr = frame_bgr.copy()
            x1_p,y1_p,x2_p,y2_p = 0,0,w_img,h_img

        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)
        face_sizes.append((face_rgb.shape[1], face_rgb.shape[0]))

        tensor = preprocess_resized(pil_face).unsqueeze(0).to(device)

        model_for_search.eval()
        with torch.no_grad():
            try:
                logits_arc, logits_aux, _ = model_for_search(tensor, None)
            except TypeError:
                logits_arc, logits_aux, _ = model_for_search(tensor)
        prob = float(torch.sigmoid(logits_aux).item())
        frame_scores.append(prob)

        model_for_search.zero_grad()
        try:
            logits_arc2, logits_aux2, _ = model_for_search(tensor, None)
        except TypeError:
            logits_arc2, logits_aux2, _ = model_for_search(tensor)
        cams = grad_multi.compute(tensor, logits_aux2, method=method)

        cam_list_resized = []
        for lname, cam in cams.items():
            cam_resized_face = cv2.resize((cam * 255).astype(np.uint8), (face_rgb.shape[1], face_rgb.shape[0])).astype(np.float32) / 255.0
            cam_list_resized.append(cam_resized_face)
        if len(cam_list_resized) == 0:
            combined_cam_face = np.zeros((face_rgb.shape[0], face_rgb.shape[1]), dtype=np.float32)
        else:
            combined_cam_face = np.mean(np.stack(cam_list_resized, axis=0), axis=0)

        _, heat_rgb_comb, overlay_comb = make_overlay_array(pil_face, combined_cam_face, alpha=0.45)
        heatmap_frames.append(overlay_comb)

        per_layer_display = []
        for lname in layer_names:
            cam = cams.get(lname, None)
            cam_resized = np.zeros((face_rgb.shape[0], face_rgb.shape[1]), dtype=np.float32) if cam is None else cv2.resize((cam * 255).astype(np.uint8), (face_rgb.shape[1], face_rgb.shape[0])).astype(np.float32) / 255.0

            h_face, w_face = cam_resized.shape[:2]
            total_mass = cam_resized.sum() + 1e-8
            ys, xs = np.mgrid[0:h_face, 0:w_face]
            cx = (cam_resized * xs).sum() / total_mass
            cy = (cam_resized * ys).sum() / total_mass
            cx_n = cx / float(max(1, w_face-1))
            cy_n = cy / float(max(1, h_face-1))
            per_layer_centroids[lname].append((cx_n, cy_n))

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
            "bbox": [int(x1_p),int(y1_p),int(x2_p),int(y2_p)],
            "combined_cam": combined_cam_face
        })

    cap.release()
    grad_multi.remove_hooks()

    # --- same analyses as before (video_score, per-layer region summary, drifts, etc.) ---
    if len(frame_scores):
        video_score = float(np.mean(frame_scores))
        report["video_level_score"] = video_score
        report["final_prediction"] = "Fake" if video_score > threshold else "Real"
    else:
        report["video_level_score"] = 0.0
        report["final_prediction"] = "N/A"

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
    # Keep computing for standard regions, but we'll only display top-2 later.
    region_names_all = ["left_eye", "right_eye", "mouth", "nose"]
    per_region_centroid_ts: Dict[str, List[Tuple[float,float]]] = {r: [] for r in region_names_all}
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
        for r in region_names_all:
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

    # Heatmap MP4
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
        heatmap_video_path = os.path.join(os.getcwd(), heatmap_name + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(heatmap_video_path, fourcc, float(heatmap_fps), (max_w, max_h))
        for fr in standardized_frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()

        # Streamlit display
        try:
            st.video(heatmap_video_path)
        except Exception:
            print("Video created at:", heatmap_video_path)

    report["layers"] = layer_names
    report["per_layer_region_summary"] = per_layer_region_summary
    report["overall_region_means"] = overall_region_means
    report["top_region_overall"] = top_region
    report["per_layer_drift"] = per_layer_drift
    report["combined_attention_drift"] = combined_drift
    report["per_region_drift"] = per_region_drift
    report["num_sampled_frames"] = len(display_rows)
    report["heatmap_video_path"] = heatmap_video_path

    # Determine top-2 regions by overall_region_means for focused drift display
    sorted_regions = sorted(overall_region_means.items(), key=lambda x: x[1], reverse=True) if overall_region_means else []
    top2_regions = [r for r,_ in sorted_regions[:2]]
    # fallback if less than 2 available
    if len(top2_regions) == 0:
        top2_regions = region_names_all[:2]
    elif len(top2_regions) == 1:
        top2_regions = top2_regions + [r for r in region_names_all if r != top2_regions[0]][:1]

    # Display top frames grid using st.pyplot
    top_rows = sorted(display_rows, key=lambda x: x['prob'], reverse=True)[:max_display]
    n_show = len(top_rows)
    if n_show == 0:
        st.info("No frames to display.")
        return report

    num_cols = 1 + len(layer_names)
    fig_w = 3 * num_cols
    fig_h = 3 * n_show
    fig = plt.figure(figsize=(fig_w, fig_h))
    for r_i, row in enumerate(top_rows):
        for c_i in range(num_cols):
            ax = fig.add_subplot(n_show, num_cols, r_i * num_cols + c_i + 1)
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
    fig.suptitle(f"Top {n_show} frames — multilayer {method} overlays (video score {report['video_level_score']:.3f} -> {report['final_prediction']})", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    st.pyplot(fig)
    plt.close(fig)

    # Best frame per-layer thumbnails
    best = top_rows[0]
    n_layers = len(layer_names)
    fig = plt.figure(figsize=(3*(n_layers+1), 4))
    ax = fig.add_subplot(1, n_layers+1, 1)
    ax.imshow(best['per_layer'][0]['orig_np']); ax.axis('off'); ax.set_title(f"Best Frame {best['frame_index']} orig (p={best['prob']:.3f})")
    for i, layer_entry in enumerate(best['per_layer']):
        ax = fig.add_subplot(1, n_layers+1, i+2)
        ax.imshow(layer_entry['overlay_np']); ax.axis('off'); ax.set_title(layer_entry['layer'])
    fig.suptitle("Best frame: original + per-layer overlays", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    st.pyplot(fig)
    plt.close(fig)

    # Temporal region plots (averaged across layers)
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
    if regions:
        fig = plt.figure(figsize=(10,4 + 0.3*len(regions)))
        for r in regions:
            plt.plot(range(len(combined_region_ts[r])), combined_region_ts[r], label=r)
        plt.xlabel("sampled frame index (temporal order)")
        plt.ylabel("mean CAM intensity")
        plt.title("Region CAM intensity over sampled frames (averaged across layers)")
        plt.legend(bbox_to_anchor=(1.01, 1.0))
        plt.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Per-region centroid drift bar - show only top-2 regions
    fig = plt.figure(figsize=(8,4))
    names = [r for r in top2_regions]
    values = [per_region_drift.get(n, 0.0) for n in names]
    plt.bar(names, values)
    plt.title("Per-region centroid path length (drift across sampled frames) — top 2 regions")
    plt.ylabel("path length (normalized coords)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Per-region centroid paths & 2D path overlay for top-2 regions
    for r in top2_regions:
        pts = per_region_centroid_ts.get(r, [])
        if len(pts) == 0:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fig = plt.figure(figsize=(6,4))
        plt.plot(range(len(xs)), xs, label=f"{r} x (norm)")
        plt.plot(range(len(ys)), ys, label=f"{r} y (norm)")
        plt.xlabel("sampled frame index")
        plt.ylabel("normalized centroid coord")
        plt.title(f"Centroid coordinates over time — region: {r} (drift={per_region_drift.get(r,0.0):.3f})")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # 2D path on best face
        best_face_np = best['per_layer'][0]['orig_np']
        h_b, w_b = best_face_np.shape[:2]
        pts_px = [(int(x*w_b), int(y*h_b)) for (x,y) in pts]
        fig = plt.figure(figsize=(4,4))
        plt.imshow(best_face_np)
        xs_px = [p[0] for p in pts_px]; ys_px = [p[1] for p in pts_px]
        plt.plot(xs_px, ys_px, '-o', color='cyan', markersize=6)
        for i,(xx,yy) in enumerate(pts_px):
            plt.text(xx+2, yy+2, str(i), color='yellow', fontsize=8)
        plt.title(f"{r} centroid path on best face")
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Combined centroid path overlay
    best_face_np = best['per_layer'][0]['orig_np']
    h_b, w_b = best_face_np.shape[:2]
    pts_px = [(int(x*w_b), int(y*h_b)) for (x,y) in combined_centroid_path]
    fig = plt.figure(figsize=(6,6))
    plt.imshow(best_face_np)
    xs = [p[0] for p in pts_px]; ys = [p[1] for p in pts_px]
    plt.scatter(xs, ys, c='yellow', s=60, edgecolors='black')
    for i in range(len(pts_px)-1):
        x0,y0 = pts_px[i]; x1,y1 = pts_px[i+1]
        plt.arrow(x0, y0, x1-x0, y1-y0, color='cyan', head_width=4, length_includes_head=True, alpha=0.8)
    plt.title(f"Combined centroid path (attention drift = {combined_drift:.3f})")
    plt.axis('off')
    st.pyplot(fig)
    plt.close(fig)

    # Print stability summary
    st.write("Region importance (avg across layers) and temporal stability (1 - cov):")
    for reg, meanv in sorted(overall_region_means.items(), key=lambda x: x[1], reverse=True):
        stabilities = []
        for lname in per_layer_region_summary.keys():
            sdict = per_layer_region_summary[lname].get(reg, None)
            if sdict:
                stabilities.append(sdict["stability"])
        stability_avg = float(np.mean(stabilities)) if stabilities else 0.0
        st.write(f"  {reg:12s} mean={meanv:.3f}  stability={stability_avg:.3f}")
    if top_region:
        st.write(f"Most important region overall: {top_region}")

    if report.get("heatmap_video_path"):
        st.write(f"Temporal heatmap video created at: {report['heatmap_video_path']}")
        st.info("Use the Download button in the main app to download the heatmap video if needed.")

    return report

# ---------------------------
# === END: Grad-CAM & Prediction block ===
# ---------------------------

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Video + GradCAM Analyzer", layout="wide")
st.title("Video-level Inference & Multi-layer Grad-CAM Analyzer")

# Sidebar controls
st.sidebar.header("Configuration")
MODEL_PATH = st.sidebar.text_input("Model path", value=MODEL_PATH)
device_choice = st.sidebar.selectbox("Device", options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], index=0)
DEVICE = "cuda" if device_choice == "cuda" and torch.cuda.is_available() else "cpu"

threshold = st.sidebar.slider("Threshold (video classification)", 0.0, 1.0, 0.5, 0.01)
num_frames = st.sidebar.slider("Number of frames to sample", 5, 50, 16, 1)
gradcam_method = st.sidebar.selectbox("Grad-CAM method", options=["gradcam", "gradcam++"], index=1)
heatmap_fps = st.sidebar.number_input("Heatmap video FPS", 1, 30, 6, 1)

st.sidebar.markdown("---")
st.sidebar.info("Upload video(s) in the main area. The app will run frame sampling, inference, and Grad-CAM.")

# cache the model loading (load once)
@st.cache_resource
def load_model_cached(path: str, device: str):
    if not os.path.exists(path):
        st.error(f"Model path does not exist: {path}")
        return None
    try:
        model = init_model_from_checkpoint(path, device=device, backbone_builder=build_efficientnetv2m_backbone, map_location=device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Show code snippets (source) for inference and grad-cam
def get_source_safe(func):
    try:
        return inspect.getsource(func)
    except Exception:
        return "Source not available."

# Upload area
st.header("Upload videos")
uploaded_files = st.file_uploader("Choose video files", accept_multiple_files=True, type=["mp4","mov","avi","mkv"])

if not uploaded_files:
    st.info("Upload one or more videos to begin analysis.")
else:
    model_loaded = None
    # Lazy load model only when user presses a button
    if st.sidebar.button("Load model"):
        with st.spinner("Loading model..."):
            model_loaded = load_model_cached(MODEL_PATH, DEVICE)
        if model_loaded:
            st.sidebar.success("Model loaded into memory.")
        else:
            st.sidebar.error("Failed to load model. Check model path and environment.")

    # We'll preserve model in session_state for multiple files
    if "model" not in st.session_state:
        st.session_state["model"] = None
    if st.session_state["model"] is None and model_loaded:
        st.session_state["model"] = model_loaded

    if st.session_state["model"] is None:
        st.warning("Model not loaded. Click 'Load model' in the sidebar to load the model before running analysis.")
    else:
        st.success("Model ready for inference.")

    # Face detector: use MediaPipe Face Detection (preferred).
    mp_face_detector = None
    MP_FACE_DET_AVAILABLE = False
    if 'mp' in globals():
        try:
            mp_face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4
            )
            MP_FACE_DET_AVAILABLE = True
        except Exception:
            MP_FACE_DET_AVAILABLE = False

    if not MP_FACE_DET_AVAILABLE:
        st.warning("MediaPipe Face Detection not available — the app will use OpenCV Haar cascade as fallback.")


    # Process each uploaded file
    results_all = []
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"Video: {uploaded_file.name}")
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.getbuffer())
        tfile.flush()
        tfile.close()
        tmp_video_path = tfile.name

        st.video(tmp_video_path)

        cols = st.columns([2,1])
        with cols[1]:
            st.write("Controls")
            local_num_frames = st.number_input(f"Frames to sample for {uploaded_file.name}", min_value=1, max_value=200, value=num_frames, key=f"nf_{uploaded_file.name}")
            local_threshold = st.slider(f"Threshold for {uploaded_file.name}", 0.0, 1.0, float(threshold), 0.01, key=f"th_{uploaded_file.name}")
            run_button = st.button(f"Run analysis for {uploaded_file.name}")

        if run_button:
            st.info("Running inference and Grad-CAM. This may take a while depending on video length and sample size.")
            # Prepare output temp files
            annotated_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            heatmap_tmpname = os.path.join(tempfile.gettempdir(), f"heatmap_{int(time.time())}.mp4")

            # Run prediction pass (annotated video + frame-level + video-level)
            with st.spinner("Running prediction and saving annotated video..."):
                # Use model in session_state but ensure it's on CPU for init_model_from_checkpoint outputs,
                # the process_and_save_video_2 will use .to(device) when appropriate.
                model_for_pred = st.session_state["model"]
                # If the cached model was loaded with init_model_from_checkpoint (already .to(device)), we need to move it to device
                try:
                    model_for_pred.to(DEVICE)
                except Exception:
                    pass

                pred_result = process_and_save_video_2(model_for_pred,mp_face_detector, tmp_video_path, annotated_tmp.name, DEVICE, local_threshold, local_num_frames)

            if pred_result is None:
                st.error("Prediction failed for this video.")
                continue

            # Run Grad-CAM analysis (this function loads its own copy of the model via init_model_from_checkpoint)
            with st.spinner("Running multi-layer Grad-CAM..."):
                try:
                    gradcam_report = run_multilayer_gradcam_display(
                        checkpoint_path=MODEL_PATH,
                        video_path=tmp_video_path,
                        backbone_builder=build_efficientnetv2m_backbone,
                        device=DEVICE,
                        num_sampled_frames=local_num_frames,
                        preprocess=DEFAULT_PREPROCESS,
                        threshold=local_threshold,
                        method=gradcam_method,
                        layers_to_use=['features.7.3.block.3.0', 'features.7.4.block.3.0', 'features.8.0'],
                        use_mediapipe=True,
                        max_display=4,
                        heatmap_fps=int(heatmap_fps),
                        heatmap_name=os.path.splitext(os.path.basename(heatmap_tmpname))[0],
                        target_size=384,
                        face_padding=0.35
                    )
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
                    gradcam_report = {"error": str(e)}

            # Present results
            st.subheader("Results & Statistics")
            st.write("Video-level score (prediction):", pred_result["video_level_score"])
            st.write("Final decision (threshold):", pred_result["final_prediction"])
            st.write("Number of sampled frames:", len(pred_result["frame_scores"]))
            st.dataframe(pd.DataFrame(pred_result["frame_scores"]))

            # Show annotated video
            st.markdown("**Annotated video (sampled frames annotated):**")
            st.video(pred_result["annotated_video"])
            with open(pred_result["annotated_video"], "rb") as f:
                st.download_button("Download annotated video", data=f, file_name=f"{Path(uploaded_file.name).stem}_annotated.mp4", mime="video/mp4")

            # Show Grad-CAM heatmap video (if created)
            heatmap_path = gradcam_report.get("heatmap_video_path") if isinstance(gradcam_report, dict) else None
            if heatmap_path and os.path.exists(heatmap_path):
                st.markdown("**Grad-CAM heatmap video:**")
                st.video(heatmap_path)
                with open(heatmap_path, "rb") as f:
                    st.download_button("Download heatmap video", data=f, file_name=f"{Path(uploaded_file.name).stem}_heatmap.mp4", mime="video/mp4")
            else:
                st.info("Heatmap video not generated or not available.")

            # Show per-layer thumbnails (top frames)
            try:
                if isinstance(gradcam_report, dict):
                    frames = gradcam_report.get("frames", [])
                    if frames:
                        st.markdown("### Representative Grad-CAM frames")
                        # show up to 4 frames
                        n_show = min(4, len(frames))
                        cols_small = st.columns(n_show)
                        for i in range(n_show):
                            entry = frames[i]
                            img = entry["per_layer"][0]["orig_np"]
                            cols_small[i].image(img, use_column_width=True, caption=f"Frame {entry['frame_index']} (p={entry['prob']:.2f})")
            except Exception:
                pass

            # Save a JSON report for download summarizing both pred_result and gradcam_report
            combined_report = {
                "video_file": uploaded_file.name,
                "prediction": pred_result,
                "gradcam_report": gradcam_report
            }
            # (Removed JSON report to avoid serialization issues — videos and heatmaps remain downloadable above)
            st.success("Analysis complete. Use the Download buttons above to get the annotated and heatmap videos.")

            # Show code blocks (in expanders)
            st.markdown("### Code used (inference & Grad-CAM)")
            with st.expander("Show inference code (process_and_save_video_2)"):
                st.code(get_source_safe(process_and_save_video_2), language="python")
            with st.expander("Show Grad-CAM code (run_multilayer_gradcam_display)"):
                st.code(get_source_safe(run_multilayer_gradcam_display), language="python")

        # cleanup temp file from upload after processing or if not processed, keep for reuse
        # we do not delete as downloads reference it; user can restart to clear cache

# End of app.py
