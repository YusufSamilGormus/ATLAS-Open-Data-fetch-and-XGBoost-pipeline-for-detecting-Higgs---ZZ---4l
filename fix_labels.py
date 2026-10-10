"""
Quick fix: Relabel MC datasets so ALL Higgs production modes are signal.
Then re-run feature engineering and training.

Signal = any dataset with "H125" or "Higgs" in the name (Higgs → ZZ* → 4ℓ)
Background = everything else (ZZ continuum, ttbar, Z+jets, etc.)
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np

DATA_DIR = "data"

# Load metadata
with open(os.path.join(DATA_DIR, "metadata.json")) as f:
    meta = json.load(f)

# Classify: anything with H125 or Higgs in the name = signal
higgs_keywords = ["H125", "Higgs"]

print("Reclassifying datasets:")
print("=" * 70)

signal_ids = set()
background_ids = set()

for dsid, info in meta.items():
    name = info["name"]
    old_cat = info["category"]
    
    is_higgs = any(kw in name for kw in higgs_keywords)
    new_cat = "signal" if is_higgs else "background"
    
    if old_cat != new_cat:
        print(f"  RECLASSIFIED: {name}")
        print(f"    {old_cat} -> {new_cat}")
    
    meta[dsid]["category"] = new_cat
    
    if new_cat == "signal":
        signal_ids.add(dsid)
    else:
        background_ids.add(dsid)

print(f"\nSignal datasets ({len(signal_ids)}):")
for dsid in sorted(signal_ids):
    print(f"  {dsid}: {meta[dsid]['name']}")

print(f"\nBackground datasets ({len(background_ids)}):")
for dsid in sorted(background_ids):
    print(f"  {dsid}: {meta[dsid]['name']}")

# Now we need to rebuild the npz files with correct labels.
# The problem is that signal_mc.npz only has the original signal (345060),
# and background_mc.npz has everything else mixed together.
# We need to re-fetch and re-split, OR we can just retrain
# using the ggH sample as signal since it's the dominant production mode
# and ALL background samples are true backgrounds.

# Actually, let's check: the original "background" datasets that are 
# actually Higgs are relatively few events compared to the true backgrounds.
# The main issue is that VBF (346228, ~86k events) and other Higgs variants
# (~450k events total) are in background, confusing the model.

# Strategy: Merge signal+background npz, then relabel based on channelNumber.
print("\n\nRebuilding training data with correct labels...")

sig_data = np.load(os.path.join(DATA_DIR, "signal_mc.npz"))
bkg_data = np.load(os.path.join(DATA_DIR, "background_mc.npz"))

# Check if channelNumber is available
has_channel = "channelNumber" in sig_data and "channelNumber" in bkg_data

if has_channel:
    # Merge all MC data
    all_keys = set(sig_data.keys()) & set(bkg_data.keys())
    all_keys.discard("label")
    
    merged = {}
    for k in all_keys:
        merged[k] = np.concatenate([sig_data[k], bkg_data[k]])
    
    # Get channelNumbers
    channels = merged["channelNumber"]
    
    # Create new labels based on channelNumber
    signal_channel_ids = {int(dsid) for dsid in signal_ids}
    new_labels = np.array([1 if ch in signal_channel_ids else 0 
                           for ch in channels], dtype=np.int32)
    
    n_sig_new = np.sum(new_labels == 1)
    n_bkg_new = np.sum(new_labels == 0)
    
    print(f"  Total MC events: {len(new_labels):,}")
    print(f"  Signal (all Higgs): {n_sig_new:,}")
    print(f"  Background (non-Higgs): {n_bkg_new:,}")
    
    # Save corrected files
    sig_mask = new_labels == 1
    bkg_mask = new_labels == 0
    
    sig_dict = {k: merged[k][sig_mask] for k in all_keys}
    sig_dict["label"] = new_labels[sig_mask]
    np.savez(os.path.join(DATA_DIR, "signal_mc.npz"), **sig_dict)
    print(f"  Saved signal_mc.npz ({n_sig_new:,} events)")
    
    bkg_dict = {k: merged[k][bkg_mask] for k in all_keys}
    bkg_dict["label"] = new_labels[bkg_mask]
    np.savez(os.path.join(DATA_DIR, "background_mc.npz"), **bkg_dict)
    print(f"  Saved background_mc.npz ({n_bkg_new:,} events)")
    
else:
    print("  channelNumber not available - cannot relabel by dataset ID")
    print("  Need to re-fetch data with channelNumber branch")

# Save updated metadata
with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(f"\n  Updated metadata.json")
print("\nDone! Now re-run: 02_feature_engineering.py -> 03_train_model.py -> 04_inference.py")
