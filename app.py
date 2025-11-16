# Multi-layer Grad-CAM Video Inspector
# Deepfake Detection with Explainable AI
import os
import torch

# Set PyTorch threading configuration BEFORE any other PyTorch operations
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
except RuntimeError:
    # Threads already set, ignore
    pass

# Set environment variables early
os.environ["OMP_NUM_THREADS"] = "4"

import streamlit as st
from pathlib import Path
import tempfile
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

# Configure Streamlit page - MUST be first Streamlit command
st.set_page_config(
    page_title="Multi-layer Grad-CAM Video Inspector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import model handler for deployment
from model_handler import get_model_path, show_model_info, handle_missing_model

# ---- IMPORTANT: the big script you pasted should be saved as gradcam_utils.py in the repo root.
# It must expose:
#   - run_multilayer_gradcam_display(...)
#   - build_efficientnetv2m_backbone
#   - DEFAULT_PREPROCESS (optional)
#   - init_model_from_checkpoint
from gradcam_utils import (
    run_multilayer_gradcam_display,
    build_efficientnetv2m_backbone,
    DEFAULT_PREPROCESS,
    init_model_from_checkpoint,
)

# -------------------- Constants and Setup --------------------
IMG_SIZE = 224                          # fixed constant: image size to use for preprocessing
CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "gradcam_streamlit"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Dynamic model path handling for deployment
@st.cache_data
def get_app_info():
    """Get application information for deployment"""
    return {
        "app_name": "Multi-layer Grad-CAM Video Inspector",
        "version": "1.0.0",
        "description": "Deepfake detection using multi-layer Grad-CAM analysis",
        "author": "Deepfake Detection System"
    }

# -------------------- Utility functions --------------------
def override_preprocess_for_imgsize(img_size: int):
    # create a torchvision-like preprocess pipeline that matches requested IMG_SIZE (fixed to 224)
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return preprocess

@st.cache_resource(show_spinner=False)
def load_model_cached(checkpoint_path: str, device: str = "cpu"):
    """
    Use the init_model_from_checkpoint from gradcam_utils to load and cache the model.
    If checkpoint_path is a URL it will be downloaded once to the cache.
    """
    if not checkpoint_path:
        return None
        
    # handle remote checkpoint URL
    if checkpoint_path.startswith("http://") or checkpoint_path.startswith("https://"):
        # Create cache path based on the checkpoint filename
        model_filename = Path(checkpoint_path).name or "model.pth"
        dest = CACHE_DIR / model_filename
        
        if not dest.exists():
            import requests
            dest_tmp = dest.with_suffix(".tmp")
            with st.spinner("Downloading checkpoint..."):
                r = requests.get(checkpoint_path, stream=True)
                r.raise_for_status()
                with open(dest_tmp, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
            dest_tmp.rename(dest)
        checkpoint_path = str(dest)

    # load model using provided initializer (this handles state_dict wrappers)
    model = init_model_from_checkpoint(checkpoint_path, device=device, backbone_builder=build_efficientnetv2m_backbone,
                                       embedding_dim=512, dropout_rate=0.5, map_location=device)
    return model

def save_uploaded_file_to_temp(uploaded_file) -> str:
    tmpdir = tempfile.mkdtemp(prefix="gradcam_vid_")
    target = os.path.join(tmpdir, uploaded_file.name)
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target

def show_top_frames_grid(report, max_display=4):
    """Enhanced top frames display with comprehensive analysis"""
    frames = report.get("frames", [])
    if len(frames) == 0:
        st.info("No frames to display in top frames. Try increasing sampled frames or uploading a different video.")
        return
    
    # Sort frames by probability (highest first)
    top_rows = sorted(frames, key=lambda x: x.get('prob', 0), reverse=True)[:max_display]
    if len(top_rows) == 0:
        st.info("No frames available for display.")
        return
    
    layers = report.get("layers", [])
    if len(layers) == 0:
        st.warning("No layers found in analysis.")
        return
    
    num_cols = 1 + len(layers)  # Original + per-layer overlays
    fig_w = 4 * num_cols
    fig_h = 4 * len(top_rows)
    
    fig, axes = plt.subplots(len(top_rows), num_cols, figsize=(fig_w, fig_h))
    if len(top_rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for r_i, row in enumerate(top_rows):
        per_layer = row.get('per_layer', [])
        if len(per_layer) == 0:
            continue
            
        for c_i in range(num_cols):
            ax = axes[r_i, c_i]
            ax.axis('off')
            
            try:
                if c_i == 0:
                    # Original frame
                    img = per_layer[0]['orig_np']
                    ax.imshow(img)
                    prob = row.get('prob', 0)
                    frame_idx = row.get('frame_index', 0)
                    ax.set_title(f"Frame {frame_idx}\n(prob={prob:.3f})", fontsize=10)
                else:
                    # Layer overlay
                    if c_i - 1 < len(per_layer):
                        layer_entry = per_layer[c_i - 1]
                        overlay = layer_entry.get('overlay_np')
                        if overlay is not None:
                            ax.imshow(overlay)
                            layer_name = layer_entry.get('layer', f'Layer {c_i}')
                            # Truncate long layer names
                            display_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
                            ax.set_title(f"{display_name}", fontsize=9)
                        else:
                            ax.text(0.5, 0.5, "No overlay", ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)[:20]}", ha='center', va='center', 
                       transform=ax.transAxes, color='red', fontsize=8)
    
    method = report.get("method", "gradcam")
    video_score = report.get("video_level_score", 0.0)
    prediction = report.get("final_prediction", "Unknown")
    
    plt.suptitle(f"Top {len(top_rows)} Suspicious Frames - {method.upper()} Analysis\n"
                f"Video Score: {video_score:.3f} → {prediction}", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)
    plt.close()

def show_detailed_analysis(report):
    """Show comprehensive analysis including drift, regions, and temporal patterns"""
    frames = report.get("frames", [])
    layers = report.get("layers", [])
    
    if not frames or not layers:
        st.warning("Insufficient data for detailed analysis.")
        return
    
    # 1. Best Frame Analysis
    st.subheader("🔍 Best Frame Detailed Analysis")
    best_frame = max(frames, key=lambda x: x.get('prob', 0))
    per_layer = best_frame.get('per_layer', [])
    
    if per_layer:
        n_layers = len(per_layer)
        cols = st.columns(min(n_layers + 1, 4))  # Limit columns for display
        
        # Original frame
        with cols[0]:
            st.write("**Original Frame**")
            orig_img = per_layer[0]['orig_np']
            st.image(orig_img, caption=f"Frame {best_frame.get('frame_index', 0)}", use_column_width=True)
            st.metric("Fake Probability", f"{best_frame.get('prob', 0):.3f}")
        
        # Per-layer overlays
        for i, layer_data in enumerate(per_layer[:3]):  # Show first 3 layers
            if i + 1 < len(cols):
                with cols[i + 1]:
                    layer_name = layer_data.get('layer', f'Layer {i}')
                    display_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
                    st.write(f"**{display_name}**")
                    overlay = layer_data.get('overlay_np')
                    if overlay is not None:
                        st.image(overlay, caption=f"Grad-CAM Overlay", use_column_width=True)
                        
                        # Region statistics
                        region_stats = layer_data.get('region_stats', {})
                        if region_stats:
                            st.write("**Region Activation:**")
                            for region, value in sorted(region_stats.items(), key=lambda x: x[1], reverse=True)[:3]:
                                st.write(f"• {region}: {value:.3f}")
    
    # 2. Temporal Region Analysis
    st.subheader("📈 Temporal Region Analysis")
    
    # Collect region data across all frames
    region_names = ["left_eye", "right_eye", "mouth", "nose", "left_cheek", "right_cheek", "forehead"]
    region_timeseries = {region: [] for region in region_names}
    
    for frame in frames:
        per_layer = frame.get('per_layer', [])
        if per_layer:
            # Average across layers for each region
            frame_regions = {region: [] for region in region_names}
            for layer_data in per_layer:
                region_stats = layer_data.get('region_stats', {})
                for region in region_names:
                    if region in region_stats:
                        frame_regions[region].append(region_stats[region])
            
            # Compute average for this frame
            for region in region_names:
                if frame_regions[region]:
                    region_timeseries[region].append(np.mean(frame_regions[region]))
                else:
                    region_timeseries[region].append(0.0)
    
    # Plot temporal evolution
    if any(len(values) > 0 for values in region_timeseries.values()):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        frame_indices = range(len(frames))
        for region, values in region_timeseries.items():
            if len(values) == len(frame_indices) and max(values) > 0.01:  # Only plot if there's activity
                ax.plot(frame_indices, values, marker='o', label=region, linewidth=2)
        
        ax.set_xlabel("Frame Index", fontsize=12)
        ax.set_ylabel("Average CAM Intensity", fontsize=12)
        ax.set_title("Temporal Evolution of Regional Attention", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # 3. Per-Region Drift Analysis
    st.subheader("🎯 Attention Drift Analysis")
    per_region_drift = report.get("per_region_drift", {})
    
    if per_region_drift:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Drift bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            regions = list(per_region_drift.keys())
            drift_values = [per_region_drift[r] for r in regions]
            
            bars = ax.bar(regions, drift_values, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.set_ylabel("Drift (Normalized Path Length)", fontsize=12)
            ax.set_title("Attention Centroid Drift by Region", fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, drift_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("**Drift Analysis Summary:**")
            sorted_drift = sorted(per_region_drift.items(), key=lambda x: x[1], reverse=True)
            for region, drift in sorted_drift:
                stability = "High" if drift < 0.1 else "Medium" if drift < 0.2 else "Low"
                color = "🟢" if stability == "High" else "🟡" if stability == "Medium" else "🔴"
                st.write(f"{color} **{region.replace('_', ' ').title()}**: {drift:.3f} ({stability} stability)")
    
    # 4. Layer-wise Analysis
    st.subheader("🧠 Layer-wise Activation Analysis")
    
    if len(layers) > 0 and frames:
        # Compute average activation per layer across all frames
        layer_activations = {}
        for layer in layers:
            activations = []
            for frame in frames:
                per_layer = frame.get('per_layer', [])
                for layer_data in per_layer:
                    if layer_data.get('layer') == layer:
                        region_stats = layer_data.get('region_stats', {})
                        if region_stats:
                            avg_activation = np.mean(list(region_stats.values()))
                            activations.append(avg_activation)
                        break
            if activations:
                layer_activations[layer] = np.mean(activations)
        
        if layer_activations:
            # Plot layer activations
            fig, ax = plt.subplots(figsize=(10, 4))
            layer_names = [name.split('.')[-1] for name in layer_activations.keys()]
            activations = list(layer_activations.values())
            
            bars = ax.bar(range(len(layer_names)), activations, color='lightcoral', alpha=0.7)
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Average Activation", fontsize=12)
            ax.set_title("Layer-wise Average Activation", fontsize=14)
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, activations)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # 5. Summary Statistics
    st.subheader("📊 Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Frames Analyzed", len(frames))
        st.metric("Layers Analyzed", len(layers))
    
    with col2:
        if per_region_drift:
            max_drift_region = max(per_region_drift.items(), key=lambda x: x[1])
            st.metric("Most Unstable Region", max_drift_region[0].replace('_', ' ').title())
            st.metric("Max Drift Value", f"{max_drift_region[1]:.3f}")
    
    with col3:
        video_score = report.get("video_level_score", 0.0)
        prediction = report.get("final_prediction", "Unknown")
        confidence = "High" if abs(video_score - 0.5) > 0.3 else "Medium" if abs(video_score - 0.5) > 0.15 else "Low"
        st.metric("Confidence Level", confidence)
        st.metric("Classification", prediction)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Multi-layer Grad-CAM Video Inspector", layout="wide", initial_sidebar_state="expanded")
st.title("🎯 Multi-layer Grad-CAM + Grad-CAM++ Video Inspector")
st.markdown(
    """
    Upload a video and the app will:
    - run multi-layer Grad-CAM / Grad-CAM++ across selected convolution layers,
    - compute temporal stability, per-region drift, combined attention drift,
    - create a temporal heatmap MP4 and display it inline,
    - show top frames with overlays and many stats.

    **Note:** The heavy model is cached — first run may be slow. Preprocessing uses a fixed resize to 224×224.
    """
)

# ---- Initialize App ----
app_info = get_app_info()
st.title(f"🎯 {app_info['app_name']}")
st.caption(app_info['description'])

# Get model path using deployment handler
default_model_path = get_model_path()

# Display model info in sidebar
with st.sidebar:
    st.header("🤖 Model Status")
    has_model = show_model_info(default_model_path)
    
    if not has_model:
        st.error("⚠️ No model available - running in demo mode")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    if default_model_path:
        checkpoint_input = st.text_input("Model checkpoint path or URL", value=str(default_model_path))
    else:
        checkpoint_input = st.text_input("Model checkpoint path or URL", 
                                       placeholder="Enter model path or URL...")
        
    method = st.selectbox("Grad-CAM method", options=["gradcam++", "gradcam"], index=0)
    num_sampled_frames = st.slider("Number of sampled frames (temporal)", 4, 32, 8, 1)
    threshold = st.slider("Fake threshold (video-level score)", 0.0, 1.0, 0.84, 0.01)
    use_mediapipe = st.checkbox("Use MediaPipe face-mesh", value=True)
    
    st.caption("⚡ Threading configured automatically for optimal performance.")
    st.markdown("---")
    st.markdown("**Advanced**")
    layers_select = st.text_input("Layers to use (comma-separated substrings) — leave empty to auto", value="")
    max_display = st.slider("Top frames to show", 1, 8, 4)
    heatmap_fps = st.number_input("Heatmap video FPS", min_value=1, max_value=30, value=6)
    st.markdown("---")
    if st.button("Show README hints"):
        st.info(
            "• If model is large host externally (S3/GDrive/GitHub releases).  \n"
            "• If you include dlib in requirements it may fail on hosted servers — use OpenCV Haar cascades instead.  \n"
            "• For Streamlit Cloud ensure mediapipe & ffmpeg availability or disable them."
        )

# ---- Upload area & example ----
st.sidebar.markdown("---")
st.sidebar.header("Input video")
uploaded = st.file_uploader("Upload a video (mp4/mov/avi) or select example below", type=["mp4","mov","avi"], accept_multiple_files=False)

col1, col2 = st.columns([2,1])
with col2:
    st.markdown("**Quick actions**")
    if st.button("Use example sample video"):
        example_remote = "https://github.com/your/repo/raw/main/examples/sample.mp4"
        uploaded = None
        try:
            import requests
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            with st.spinner("Downloading example..."):
                r = requests.get(example_remote, stream=True, timeout=10)
                r.raise_for_status()
                for chunk in r.iter_content(8192):
                    if chunk:
                        tmp.write(chunk)
            tmp.flush()
            tmp.close()
            uploaded_temp_path = tmp.name
            st.success("Example downloaded")
            st.session_state["example_path"] = uploaded_temp_path
        except Exception:
            st.error("Failed to download example. Provide your own video.")
            st.session_state["example_path"] = None

with col1:
    st.markdown("Drop video file here or use example.")

# ---- Start processing button ----
run_it = st.button("Run analysis")

# ---- When run clicked: prepare inputs and call the heavy function ----
if run_it:
    # determine video path
    video_path = None
    if uploaded is not None:
        video_path = save_uploaded_file_to_temp(uploaded)
    elif st.session_state.get("example_path", None):
        video_path = st.session_state["example_path"]
    else:
        st.error("No video provided. Upload a file or click the example button.")
        st.stop()

    # model checkpoint handling
    checkpoint_path = checkpoint_input.strip() or default_model_path
    
    if not checkpoint_path:
        handle_missing_model()

    # show runtime info
    st.info(f"🔧 Using checkpoint: `{checkpoint_path}` — will be cached after first load. Image size = {IMG_SIZE}x{IMG_SIZE}")
    st.write("⚡ Threading:", f"PyTorch configured for optimal CPU performance. OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    # fixed preprocess to IMG_SIZE=224
    preprocess = override_preprocess_for_imgsize(IMG_SIZE)

    # process layers_to_use
    layers_to_use = [s.strip() for s in layers_select.split(",") if s.strip()] or None

    # load model (cached). This is synchronous and cached by Streamlit
    try:
        with st.spinner("Loading model (cached) — this may take some time on first run..."):
            model = load_model_cached(checkpoint_path, device="cpu")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    # Run analysis — heavy compute
    start_t = time.time()
    try:
        with st.spinner("Running multi-layer Grad-CAM analysis on the video — this may take a while"):
            report = run_multilayer_gradcam_display(
                checkpoint_path=checkpoint_path,
                video_path=video_path,
                backbone_builder=build_efficientnetv2m_backbone,
                device="cpu",
                num_sampled_frames=int(num_sampled_frames),
                preprocess=preprocess,
                threshold=float(threshold),
                method=str(method),
                layers_to_use=layers_to_use,
                use_mediapipe=bool(use_mediapipe),
                max_display=int(max_display),
                heatmap_fps=int(heatmap_fps),
                heatmap_name="temporal_heatmap_streamlit"
            )
            
        # Verify we have data
        if not report:
            st.error("Analysis failed - no report generated")
            st.stop()
            
        frames = report.get("frames", [])
        if len(frames) == 0:
            st.warning("No frames were processed. This might be due to:")
            st.write("• Video format compatibility issues")
            st.write("• Model loading problems")
            st.write("• Face detection failures")
            st.write("Debug info:")
            st.write(f"• Checkpoint: {checkpoint_path}")
            st.write(f"• Video: {video_path}")
            st.write(f"• Report keys: {list(report.keys())}")
        else:
            st.success(f"✅ Successfully analyzed {len(frames)} frames!")
            
    except Exception as e:
        st.exception(e)
        st.error("Analysis failed. Please check:")
        st.write("• Video file is valid and readable")
        st.write("• Model checkpoint exists and is valid")
        st.write("• All dependencies are installed correctly")
        st.stop()

    elapsed = time.time() - start_t
    st.success(f"Analysis finished in {elapsed:.1f}s")

    # ---- Show summarized results ----
    st.header("🎯 Analysis Results")
    
    # Main metrics
    colA, colB, colC, colD = st.columns(4)
    with colA:
        video_score = report.get('video_level_score', 0.0)
        st.metric("Video-level Score", f"{video_score:.4f}")
    with colB:
        prediction = report.get("final_prediction", "N/A")
        st.metric("Final Prediction", prediction)
    with colC:
        num_frames = report.get("num_sampled_frames", 0)
        st.metric("Sampled Frames", num_frames)
    with colD:
        layers_analyzed = len(report.get("layers", []))
        st.metric("Layers Analyzed", layers_analyzed)

    # Enhanced visualizations
    st.header("🔍 Detailed Analysis")
    
    # Show comprehensive top frames analysis
    st.subheader("Top Suspicious Frames with Grad-CAM Overlays")
    try:
        show_top_frames_grid(report, max_display=int(max_display))
    except Exception as e:
        st.error(f"Failed to render top frames: {e}")
        st.write("Debug info:", str(report.get("frames", [])[:1]))  # Show first frame for debugging

    # Show detailed analysis
    try:
        show_detailed_analysis(report)
    except Exception as e:
        st.error(f"Failed to render detailed analysis: {e}")
    
    # Heatmap video
    heatmap_path = report.get("heatmap_video_path")
    if heatmap_path:
        st.subheader("🎬 Temporal Heatmap Video")
        
        # Check if file exists and has content
        if os.path.exists(heatmap_path):
            file_size = os.path.getsize(heatmap_path)
            st.write(f"📹 Video file: {os.path.basename(heatmap_path)} ({file_size:,} bytes)")
            
            if file_size > 1000:  # At least 1KB
                try:
                    # Read video file
                    with open(heatmap_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    # Display with Streamlit video component
                    st.video(video_bytes)
                    st.success("✅ Temporal heatmap video loaded!")
                    
                    # Download option
                    st.download_button(
                        "📥 Download Heatmap Video",
                        video_bytes,
                        file_name=f"temporal_heatmap_{int(time.time())}.mp4",
                        mime="video/mp4"
                    )
                    
                except Exception as e:
                    st.error(f"Error loading video: {e}")
                    st.write(f"Video path: `{heatmap_path}`")
                    st.info("Try downloading the video file to view it externally.")
            else:
                st.warning(f"⚠️ Video file is too small ({file_size} bytes) - may be corrupted")
        else:
            st.warning("⚠️ Video file was not found")
    else:
        st.info("ℹ️ No temporal heatmap video was generated")
    
    # Raw data section (collapsible)
    with st.expander("📋 Raw Analysis Report (JSON)", expanded=False):
        # Filter out large data for cleaner display
        filtered_report = {k: v for k, v in report.items() 
                          if k not in ("frames",) and not k.endswith("_path")}
        st.code(json.dumps(filtered_report, indent=2), language="json")
    
    # Download section
    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Full report download
        full_report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            "📄 Download Full Report (JSON)", 
            data=full_report_json, 
            file_name=f"gradcam_analysis_{int(time.time())}.json", 
            mime="application/json"
        )
    
    with col2:
        # Summary report
        summary = {
            "analysis_timestamp": time.time(),
            "video_score": video_score,
            "prediction": prediction,
            "confidence": "High" if abs(video_score - 0.5) > 0.3 else "Medium" if abs(video_score - 0.5) > 0.15 else "Low",
            "num_frames": num_frames,
            "layers": report.get("layers", []),
            "method": report.get("method", method),
            "per_region_drift": report.get("per_region_drift", {}),
            "top_region": report.get("top_region_overall", "N/A")
        }
        summary_json = json.dumps(summary, indent=2)
        st.download_button(
            "📊 Download Summary Report", 
            data=summary_json, 
            file_name=f"gradcam_summary_{int(time.time())}.json", 
            mime="application/json"
        )

    st.markdown("---")
    st.caption("Tip: If the checkpoint is large, host it externally (S3 / GitHub release) and paste the URL in the model checkpoint field.")
