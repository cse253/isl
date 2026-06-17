"""
run_project.py
Master runner script that executes the entire Weeks 2-5 project pipeline.
Verifies all dependencies, run steps, output results, and logs the execution.
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "baseline.yaml"
REQUIREMENTS_PATH = ROOT / "requirements.txt"
DATASET_PATH = ROOT / "datasets" / "Adjectives_1of8" / "Adjectives"
RESULTS_PATH = ROOT / "results"
LOG_FILE_PATH = RESULTS_PATH / "full_project_run_log.txt"

# ── List of Files to Verify ───────────────────────────────────────────────────
FILES_TO_VERIFY = [
    # Scripts
    "scripts/explore_dataset.py",
    "scripts/split_dataset.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/extract_pose.py",
    "scripts/train_pose.py",
    "scripts/train_fusion.py",
    "scripts/train_adaptive.py",
    "scripts/visualize_fusion_weights.py",
    "scripts/week4_compare.py",
    "scripts/run_ablation.py",
    "scripts/grad_cam.py",
    "scripts/attention_visualization.py",
    "scripts/analyze_fusion_weights.py",
    "scripts/confusion_analysis.py",
    # Models
    "models/rgb_branch.py",
    "models/pose_branch.py",
    "models/fusion.py",
    "models/adaptive_fusion.py",
    "models/adaptive_model.py",
    # Configs & metadata
    "configs/baseline.yaml",
    "requirements.txt",
]

# ── Expected Outputs to Verify after running ──────────────────────────────────
EXPECTED_OUTPUTS = [
    "results/class_distribution.png",
    "results/dataset_statistics.csv",
    "results/training_log.csv",
    "results/confusion_matrix.png",
    "results/classification_report.txt",
    "results/adaptive_training_log.csv",
    "results/fusion_weights.png",
    "results/week4_comparison.csv",
    "results/week4_comparison.png",
    "results/ablation_results.csv",
    "results/ablation_accuracy_comparison.png",
    "results/fusion_weight_analysis.csv",
    "results/fusion_weight_distribution.png",
    "results/confusion_analysis.csv",
    "results/week5_summary.txt",
    "results/gradcam",
    "results/attention_maps",
    "checkpoints",
]

# ── Required Python Packages ───────────────────────────────────────────────────
REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "mediapipe",
    "yaml",
    "tqdm"
]

# ── Sequential Pipeline Steps ──────────────────────────────────────────────────
PIPELINE_STEPS = [
    {"cmd": [sys.executable, "scripts/explore_dataset.py"], "critical": True, "desc": "Dataset Exploration"},
    {"cmd": [sys.executable, "scripts/split_dataset.py"], "critical": True, "desc": "Train/Val/Test Split"},
    {"cmd": [sys.executable, "scripts/train.py"], "critical": True, "desc": "RGB Baseline Training"},
    {"cmd": [sys.executable, "scripts/evaluate.py"], "critical": True, "desc": "RGB Baseline Evaluation"},
    {"cmd": [sys.executable, "scripts/extract_pose.py"], "critical": True, "desc": "Pose Landmark Extraction"},
    {"cmd": [sys.executable, "scripts/train_pose.py"], "critical": True, "desc": "Pose Branch Training"},
    {"cmd": [sys.executable, "scripts/train_fusion.py"], "critical": True, "desc": "Multi-Modal Fusion Training"},
    {"cmd": [sys.executable, "models/adaptive_model.py"], "critical": True, "desc": "Adaptive Model Sanity Test"},
    {"cmd": [sys.executable, "scripts/train_adaptive.py"], "critical": True, "desc": "Adaptive Fusion Training"},
    {"cmd": [sys.executable, "scripts/visualize_fusion_weights.py"], "critical": False, "desc": "Fusion Weights Plotting"},
    {"cmd": [sys.executable, "scripts/week4_compare.py"], "critical": True, "desc": "Week 4 Model Comparison"},
    {"cmd": [sys.executable, "scripts/run_ablation.py"], "critical": True, "desc": "Week 5 Ablation Study"},
    {"cmd": [sys.executable, "scripts/grad_cam.py"], "critical": False, "desc": "Grad-CAM Spatial Explainability"},
    {"cmd": [sys.executable, "scripts/attention_visualization.py"], "critical": False, "desc": "Transformer Attention Maps"},
    {"cmd": [sys.executable, "scripts/analyze_fusion_weights.py"], "critical": True, "desc": "Fusion Weight Analysis"},
    {"cmd": [sys.executable, "scripts/confusion_analysis.py"], "critical": True, "desc": "Confusion Matrix Analysis"},
]

def check_folders_and_dataset():
    """Verify directories and dataset folder exist."""
    print("=" * 60)
    print("  STAGE 1: CHECKING PROJECT DIRS & DATASET")
    print("=" * 60)
    
    # Required folders
    folders = ["configs", "datasets", "models", "scripts", "splits"]
    for f in folders:
        dir_path = ROOT / f
        if not dir_path.exists():
            print(f"  [WARN] Required folder does not exist: {f}")
        else:
            print(f"  [OK] Folder exists: {f}")
            
    # Dataset path
    if not DATASET_PATH.exists():
        print(f"\n[ERROR] Dataset not found at: {DATASET_PATH}")
        print("Please place the dataset under datasets/Adjectives_1of8/Adjectives/")
        sys.exit(1)
    else:
        print(f"  [OK] Dataset found at: {DATASET_PATH.relative_to(ROOT)}")
        
    # Check classes
    expected_classes = ["loud", "quiet", "happy", "sad", "Beautiful", "Ugly", "Deaf", "Blind"]
    found_classes = [d.name.split(". ", 1)[-1] for d in DATASET_PATH.iterdir() if d.is_dir()]
    for ec in expected_classes:
        if ec not in found_classes:
            print(f"  [WARN] Expected class folder not found: {ec}")
        else:
            print(f"  [OK] Class folder found: {ec}")
    print()

def check_python_packages():
    """Check required packages are installed."""
    print("=" * 60)
    print("  STAGE 2: CHECKING PYTHON ENVIRONMENT")
    print("=" * 60)
    missing = []
    for pkg in REQUIRED_PACKAGES:
        # Map some imports to their pip names if different
        import_name = pkg
        if pkg == "cv2":
            import_name = "cv2"
        elif pkg == "yaml":
            import_name = "yaml"
        elif pkg == "sklearn":
            import_name = "sklearn"
            
        try:
            __import__(import_name)
            print(f"  [OK] Package available: {pkg}")
        except ImportError:
            print(f"  [MISSING] Package missing: {pkg}")
            missing.append(pkg)
            
    if missing:
        print(f"\n[ERROR] Missing required packages: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    print()

def verify_files():
    """Verify presence of all project code files."""
    print("=" * 60)
    print("  STAGE 3: VERIFYING PROJECT CODE CODEBASE")
    print("=" * 60)
    missing_files = []
    for rel_path in FILES_TO_VERIFY:
        path = ROOT / rel_path
        if not path.exists():
            print(f"  MISSING: {rel_path}")
            missing_files.append(rel_path)
        else:
            print(f"  [OK] Found file: {rel_path}")
            
    if missing_files:
        print(f"\n[ERROR] Project verify failed. {len(missing_files)} file(s) missing.")
        sys.exit(1)
    print()

def run_pipeline():
    """Executes the pipeline steps sequentially."""
    print("=" * 60)
    print("  STAGE 4: EXECUTING PROJECT RUNNER PIPELINE")
    print("=" * 60)
    
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Configure non-interactive backend for all child processes
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    
    summary = []
    
    # Open log file to record full stdout and stderr
    with open(LOG_FILE_PATH, "w", encoding="utf-8") as log_f:
        log_f.write(f"Adaptive Multi-Modal Fusion Transformer - Full Project Run Log\n")
        log_f.write(f"Started: {time.asctime()}\n")
        log_f.write("=" * 80 + "\n\n")
        
        for step_idx, step in enumerate(PIPELINE_STEPS, 1):
            desc = step["desc"]
            cmd = step["cmd"]
            cmd_str = " ".join(cmd[1:]) if len(cmd) > 1 else cmd[0]
            is_critical = step["critical"]
            
            print(f"[{step_idx}/{len(PIPELINE_STEPS)}] Running {desc} ...")
            print(f"  Command: python {cmd_str}")
            
            # Warn if step uses ResNet50 on CPU
            if any(name in cmd_str for name in ["train.py", "evaluate.py", "run_ablation.py", "grad_cam.py", "attention_visualization.py", "confusion_analysis.py"]):
                print("  [WARNING] This step runs ResNet50 evaluations on CPU and may take 1-5 minutes.")
                
            log_f.write(f"\n--- STEP {step_idx}: {desc} ({cmd_str}) ---\n")
            log_f.flush()
            
            t0 = time.time()
            try:
                # Run script
                res = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False
                )
                
                duration = time.time() - t0
                log_f.write(res.stdout)
                log_f.write(f"\nDuration: {duration:.2f} seconds\n")
                log_f.write(f"Return Code: {res.returncode}\n")
                log_f.flush()
                
                if res.returncode == 0:
                    print(f"  [OK] Completed in {duration:.2f} seconds.\n")
                    summary.append({"step": desc, "status": "OK", "reason": f"Finished in {duration:.1f}s"})
                else:
                    print(f"  [FAIL] Failed with exit code {res.returncode}.\n")
                    # Capture snippet of output for fast debugging
                    err_lines = res.stdout.strip().split("\n")[-10:]
                    err_snippet = "\n".join([f"    {l}" for l in err_lines])
                    print("  Last output lines:")
                    print(err_snippet)
                    print()
                    
                    summary.append({"step": desc, "status": "FAILED", "reason": f"Exit code {res.returncode}"})
                    if is_critical:
                        print(f"[CRITICAL ERROR] Critical step '{desc}' failed. Aborting pipeline execution.")
                        log_f.write(f"\n[ABORTED] Critical step failed.\n")
                        break
                        
            except Exception as e:
                duration = time.time() - t0
                log_f.write(f"Exception during run:\n{str(e)}\n")
                log_f.flush()
                print(f"  [FAIL] Exception: {e}\n")
                summary.append({"step": desc, "status": "FAILED", "reason": str(e)})
                if is_critical:
                    print(f"[CRITICAL ERROR] Critical step '{desc}' encountered exception. Aborting pipeline.")
                    break
                    
        log_f.write("\n" + "=" * 80 + "\n")
        log_f.write(f"Finished: {time.asctime()}\n")
        
    return summary

def verify_outputs():
    """Check that output results exist."""
    print("=" * 60)
    print("  STAGE 5: VERIFYING OUTPUT RESULTS")
    print("=" * 60)
    
    missing_outputs = []
    for path_str in EXPECTED_OUTPUTS:
        path = ROOT / path_str
        if not path.exists():
            print(f"  [MISSING] Output missing: {path_str}")
            missing_outputs.append(path_str)
        else:
            size_str = ""
            if path.is_file():
                size_str = f" ({path.stat().st_size:,} bytes)"
            print(f"  [OK] Output generated: {path_str}{size_str}")
            
    print()
    return missing_outputs

def print_summary(summary, missing_outputs):
    """Prints the final success/failure summary."""
    print("=" * 60)
    print("  PROJECT RUN SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for item in summary:
        status_str = f"[{item['status']}]"
        # Pad status
        status_pad = f"{status_str:<10}"
        if item["status"] == "OK":
            print(f"  \033[92m{status_pad}\033[0m {item['step']} — {item['reason']}")
        else:
            print(f"  \033[91m{status_pad}\033[0m {item['step']} — {item['reason']}")
            all_ok = False
            
    print("-" * 60)
    if not missing_outputs:
        print("  \033[92m[OK] All expected results files are present.\033[0m")
    else:
        print(f"  \033[91m[FAILED] {len(missing_outputs)} expected results files are missing.\033[0m")
        all_ok = False
        
    print("=" * 60)
    print(f"Full pipeline log saved to: {LOG_FILE_PATH}")
    if all_ok:
        print("\033[92mSUCCESS: Entire project pipeline Weeks 2-5 executed correctly!\033[0m")
    else:
        print("\033[91mFAILURE: Some pipeline steps or output files failed validation.\033[0m")
    print("=" * 60)

if __name__ == "__main__":
    check_folders_and_dataset()
    check_python_packages()
    verify_files()
    summary = run_pipeline()
    missing_outputs = verify_outputs()
    print_summary(summary, missing_outputs)
