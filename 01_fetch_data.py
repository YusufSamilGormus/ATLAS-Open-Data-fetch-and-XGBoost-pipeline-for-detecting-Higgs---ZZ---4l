"""
Higgs Boson Classifier — Step 1: Data Fetching & Exploration
============================================================
Uses atlasopenmagic to discover ATLAS Open Data datasets (2025e-13tev-beta)
and uproot to read ROOT file branches.

This script:
  1. Connects to ATLAS Open Data and lists available datasets
  2. Fetches signal MC (ggH + VBF Higgs → ZZ* → 4ℓ) URLs
  3. Fetches background MC (ZZ, ttbar, etc.) URLs
  4. Fetches real collision data URLs
  5. Reads and inspects the TTree branch structure
  6. Saves all data locally as .npz files for subsequent steps
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

import atlasopenmagic as atom
import uproot
import awkward as ak
import numpy as np
import json
import os
import sys


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

RELEASE = "2025e-13tev-beta"
SKIM = "exactly4lep"
OUTPUT_DIR = "data"

# Branches to read from the ROOT files (2025e-13tev-beta naming)
LEPTON_BRANCHES = [
    "lep_pt", "lep_eta", "lep_phi", "lep_e",
    "lep_charge", "lep_type", "lep_isTightID",
    "lep_ptvarcone30", "lep_topoetcone20",
    "lep_d0sig", "lep_z0",
]

EVENT_BRANCHES = [
    "met", "met_phi",
]

MC_BRANCHES = [
    "mcWeight",
    "ScaleFactor_PILEUP",
    "ScaleFactor_ELE",
    "ScaleFactor_MUON",
    "ScaleFactor_LepTRIGGER",
    "xsec", "filteff", "kfac",
    "sum_of_weights", "num_events",
    "channelNumber",
]

# All branches for MC files
ALL_MC_BRANCHES = LEPTON_BRANCHES + EVENT_BRANCHES + MC_BRANCHES

# For real data, no MC-specific branches
ALL_DATA_BRANCHES = LEPTON_BRANCHES + EVENT_BRANCHES


# ═══════════════════════════════════════════════════════════════
# PHASE 1: DISCOVER DATASETS
# ═══════════════════════════════════════════════════════════════

def setup_release():
    """Set the ATLAS Open Data release and show available options."""
    print("=" * 60)
    print("PHASE 1: Setting up ATLAS Open Data release")
    print("=" * 60)

    print("\nAvailable releases:")
    releases = atom.available_releases()
    print(f"  {releases}")

    atom.set_release(RELEASE)
    print(f"\n✓ Active release set to: {RELEASE}")

    print("\nAvailable data options:")
    try:
        data_options = atom.available_datasets()
        print(f"  {data_options}")
    except Exception as e:
        data_options = f"(Could not list: {e})"
        print(f"  {data_options}")

    return releases, data_options


def discover_datasets():
    """
    Discover signal and background dataset IDs.
    We search the metadata for relevant physics processes.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Discovering datasets")
    print("=" * 60)

    # Try to get all metadata to find signal and background IDs
    # We know from ATLAS documentation:
    #   Signal:  ggH (345060), VBF (344235)
    #   Background: ZZ, ttbar, Z+jets, etc.

    datasets = {
        "signal": {},
        "background": {},
        "data": {}
    }

    # --- Signal datasets ---
    signal_ids = {
        "ggH125_ZZ4lep": "345060",
        "VBFH125_ZZ4lep": "344235",
    }

    print("\n--- Signal MC datasets ---")
    for name, dsid in signal_ids.items():
        try:
            meta = atom.get_metadata(dsid)
            if not isinstance(meta, dict):
                meta = {}
            urls = atom.get_urls(dsid, skim=SKIM, protocol='https')
            if isinstance(urls, str):
                urls = [urls]
            datasets["signal"][dsid] = {
                "name": name,
                "urls": urls if urls else [],
                "n_files": len(urls) if urls else 0,
                "metadata": {k: v for k, v in meta.items()
                             if k in ["cross_section_pb", "filter_efficiency",
                                      "k_factor", "generator"]},
            }
            print(f"  ✓ {name} (ID: {dsid}): {len(urls)} file(s)")
            print(f"    Cross-section: {meta.get('cross_section_pb', 'N/A')} pb")
        except Exception as e:
            print(f"  ✗ {name} (ID: {dsid}): {e}")

    # --- Background datasets ---
    print("\n--- Background MC datasets ---")
    print("  Discovering background datasets from metadata...")

    # Get list of all available dataset IDs
    try:
        all_datasets = atom.available_datasets()
        # all_datasets could be a list or dict
        if isinstance(all_datasets, dict):
            dataset_ids = list(all_datasets.keys())
        elif isinstance(all_datasets, list):
            dataset_ids = all_datasets
        else:
            dataset_ids = []

        print(f"  Found {len(dataset_ids)} total datasets in release")

        # Filter for background processes relevant to 4-lepton analysis
        bg_keywords = ["llll", "zee", "zmumu", "ttbar", "zz", "wz",
                        "singletop", "diboson"]

        for dsid in dataset_ids:
            dsid_str = str(dsid)
            # Skip signal datasets we already have
            if dsid_str in datasets["signal"]:
                continue
            try:
                meta = atom.get_metadata(dsid_str)
                if not isinstance(meta, dict):
                    continue
                phys_short = str(meta.get("physics_short", "")).lower()
                # Check if this is a relevant background
                if any(bg in phys_short for bg in bg_keywords):
                    try:
                        urls = atom.get_urls(dsid_str, skim=SKIM,
                                             protocol='https')
                        if urls:
                            if isinstance(urls, str):
                                urls = [urls]
                            datasets["background"][dsid_str] = {
                                "name": meta.get("physics_short", dsid_str),
                                "urls": urls,
                                "n_files": len(urls),
                                "metadata": {
                                    k: v for k, v in meta.items()
                                    if k in ["cross_section_pb",
                                             "filter_efficiency", "k_factor"]
                                },
                            }
                            print(f"  ✓ {meta.get('physics_short', dsid_str)} "
                                  f"(ID: {dsid_str}): {len(urls)} file(s)")
                    except Exception:
                        pass  # Skim not available for this dataset
            except Exception:
                continue

    except Exception as e:
        print(f"  Could not enumerate datasets: {e}")

    # --- Real collision data ---
    print("\n--- Real collision data ---")
    try:
        data_urls = atom.get_urls("data", skim=SKIM, protocol='https')
        datasets["data"]["collision"] = {
            "name": "Real pp collision data",
            "urls": data_urls,
            "n_files": len(data_urls),
        }
        print(f"  ✓ Collision data: {len(data_urls)} file(s)")
    except Exception as e:
        print(f"  ✗ Could not get data URLs: {e}")

    return datasets


# ═══════════════════════════════════════════════════════════════
# PHASE 3: READ ROOT FILES
# ═══════════════════════════════════════════════════════════════

def inspect_tree(url):
    """Open one ROOT file and print its TTree structure."""
    print(f"\n--- Inspecting: {url.split('/')[-1]} ---")
    try:
        with uproot.open(f"{url}:analysis") as tree:
            print(f"  TTree 'mini': {tree.num_entries} entries")
            print(f"  Available branches ({len(tree.keys())}):")
            for key in sorted(tree.keys()):
                branch = tree[key]
                print(f"    {key:30s}  {branch.typename}")
            return tree.keys()
    except Exception as e:
        # Try without specifying tree name
        print(f"  Could not open 'mini' tree: {e}")
        print("  Trying to list all keys in file...")
        try:
            with uproot.open(url) as f:
                print(f"  File keys: {f.keys()}")
                # Try the first key that looks like a TTree
                for key in f.keys():
                    if "TTree" in str(type(f[key])):
                        print(f"  Found TTree: {key}")
                        return f[key].keys()
        except Exception as e2:
            print(f"  Error: {e2}")
    return []


def read_root_files(urls, branches, label=None, max_files=None):
    """
    Read specified branches from multiple ROOT files.

    Args:
        urls: List of ROOT file URLs
        branches: List of branch names to read
        label: Optional label to add (1=signal, 0=background, None=data)
        max_files: Max number of files to read (None = all)

    Returns:
        awkward.Array with all events concatenated
    """
    chunks = []
    n_files = min(len(urls), max_files) if max_files else len(urls)

    for i, url in enumerate(urls[:n_files]):
        filename = url.split("/")[-1]
        try:
            with uproot.open(f"{url}:analysis") as tree:
                # Only read branches that actually exist in this file
                available = set(tree.keys())
                valid_branches = [b for b in branches if b in available]
                missing = set(branches) - available
                if missing and i == 0:
                    print(f"  Note: Branches not found: {missing}")

                chunk = tree.arrays(valid_branches)
                n_events = len(chunk)
                chunks.append(chunk)
                print(f"  [{i+1}/{n_files}] {filename}: {n_events} events")
        except Exception as e:
            print(f"  [{i+1}/{n_files}] {filename}: FAILED - {e}")
            # Try alternative tree names
            try:
                with uproot.open(url) as f:
                    keys = f.keys()
                    print(f"    Available keys: {keys}")
                    for key in keys:
                        key_clean = key.split(";")[0]
                        try:
                            tree = f[key]
                            if hasattr(tree, 'arrays'):
                                available = set(tree.keys())
                                valid_branches = [b for b in branches
                                                  if b in available]
                                chunk = tree.arrays(valid_branches)
                                chunks.append(chunk)
                                print(f"    ✓ Read from '{key_clean}': "
                                      f"{len(chunk)} events")
                                break
                        except Exception:
                            continue
            except Exception as e2:
                print(f"    Could not recover: {e2}")

    if not chunks:
        print("  WARNING: No data was read!")
        return None

    combined = ak.concatenate(chunks)
    print(f"  Total: {len(combined)} events")
    return combined


# ═══════════════════════════════════════════════════════════════
# PHASE 4: SAVE DATA
# ═══════════════════════════════════════════════════════════════

def awkward_to_numpy(data, branches):
    """Convert awkward array to dict of numpy arrays."""
    result = {}
    for branch in branches:
        if branch not in ak.fields(data):
            continue
        arr = data[branch]
        try:
            # Try converting jagged → regular (works for exactly4lep)
            result[branch] = ak.to_numpy(ak.to_regular(arr))
        except Exception:
            try:
                result[branch] = ak.to_numpy(arr)
            except Exception:
                print(f"  Warning: Could not convert branch '{branch}' "
                      f"to numpy, skipping.")
    return result


def save_dataset(data_dict, name, output_dir):
    """Save numpy arrays to a .npz file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name}.npz")
    np.savez(filepath, **data_dict)
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"  ✓ Saved {filepath} ({size_mb:.1f} MB)")
    return filepath


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Higgs Boson Classifier — Data Fetching & Exploration  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Setup
    releases, data_options = setup_release()

    # Step 2: Discover datasets
    datasets = discover_datasets()

    # Step 3: Inspect one file to see the branch structure
    print("\n" + "=" * 60)
    print("PHASE 3: Inspecting ROOT file structure")
    print("=" * 60)

    # Pick any available URL to inspect
    sample_url = None
    for category in ["signal", "background", "data"]:
        for dsid, info in datasets.get(category, {}).items():
            if info.get("urls"):
                sample_url = info["urls"][0]
                break
        if sample_url:
            break

    if sample_url:
        available_branches = inspect_tree(sample_url)
    else:
        print("  No files available to inspect!")
        available_branches = []

    # Step 4: Read all data
    print("\n" + "=" * 60)
    print("PHASE 4: Reading ROOT files")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine which branches are actually available
    mc_branches = [b for b in ALL_MC_BRANCHES if b in available_branches] \
        if available_branches else ALL_MC_BRANCHES
    data_branches = [b for b in ALL_DATA_BRANCHES if b in available_branches] \
        if available_branches else ALL_DATA_BRANCHES

    # Read signal MC
    print("\n--- Reading Signal MC ---")
    signal_arrays = {}
    for dsid, info in datasets.get("signal", {}).items():
        print(f"\n  Dataset: {info['name']} (ID: {dsid})")
        data = read_root_files(info["urls"], mc_branches)
        if data is not None:
            signal_arrays[dsid] = data

    # Read background MC
    print("\n--- Reading Background MC ---")
    background_arrays = {}
    for dsid, info in datasets.get("background", {}).items():
        print(f"\n  Dataset: {info['name']} (ID: {dsid})")
        data = read_root_files(info["urls"], mc_branches)
        if data is not None:
            background_arrays[dsid] = data

    # Read real collision data
    print("\n--- Reading Real Collision Data ---")
    real_data = None
    for dsid, info in datasets.get("data", {}).items():
        print(f"\n  Dataset: {info['name']}")
        real_data = read_root_files(info["urls"], data_branches)

    # Step 5: Convert and save
    print("\n" + "=" * 60)
    print("PHASE 5: Converting and saving data")
    print("=" * 60)

    # Combine all signal MC
    if signal_arrays:
        all_signal = ak.concatenate(list(signal_arrays.values()))
        signal_np = awkward_to_numpy(all_signal, mc_branches)
        signal_np["label"] = np.ones(len(all_signal), dtype=np.int32)
        save_dataset(signal_np, "signal_mc", OUTPUT_DIR)
        print(f"  Signal events: {len(all_signal)}")

    # Combine all background MC
    if background_arrays:
        all_bkg = ak.concatenate(list(background_arrays.values()))
        bkg_np = awkward_to_numpy(all_bkg, mc_branches)
        bkg_np["label"] = np.zeros(len(all_bkg), dtype=np.int32)
        save_dataset(bkg_np, "background_mc", OUTPUT_DIR)
        print(f"  Background events: {len(all_bkg)}")

    # Save real data
    if real_data is not None:
        data_np = awkward_to_numpy(real_data, data_branches)
        save_dataset(data_np, "collision_data", OUTPUT_DIR)
        print(f"  Real data events: {len(real_data)}")

    # Save dataset metadata for later use
    meta_to_save = {}
    for category in ["signal", "background"]:
        for dsid, info in datasets.get(category, {}).items():
            meta_to_save[dsid] = {
                "name": info["name"],
                "category": category,
                "n_files": info["n_files"],
                "metadata": info.get("metadata", {}),
            }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta_to_save, f, indent=2, default=str)
    print(f"\n  ✓ Saved metadata to {OUTPUT_DIR}/metadata.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_sig = len(all_signal) if signal_arrays else 0
    n_bkg = len(all_bkg) if background_arrays else 0
    n_data = len(real_data) if real_data is not None else 0
    print(f"  Signal MC events:     {n_sig:>8,}")
    print(f"  Background MC events: {n_bkg:>8,}")
    print(f"  Real data events:     {n_data:>8,}")
    print(f"  Output directory:     {os.path.abspath(OUTPUT_DIR)}")
    print("\n  Next step: Run 02_feature_engineering.py")


if __name__ == "__main__":
    main()
