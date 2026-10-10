"""
Higgs Boson Classifier — Step 2: Feature Engineering
=====================================================
Computes physics-informed features from raw lepton kinematics.

Key features computed:
  1. Four-lepton invariant mass (m4l) — THE Higgs discriminant
  2. Z-boson candidate pairing (Z1 on-shell, Z2 off-shell)
  3. Z1 and Z2 invariant masses (mZ1, mZ2)
  4. Angular variables (cos_theta_star, delta_phi_Z1Z2, etc.)
  5. Per-lepton kinematics (leading/subleading pT)
  6. MET-related features
  7. MC event weights for training

Units: all energies/masses already in GeV in the 2025e-13tev-beta release.
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DATA_DIR = "data"
OUTPUT_DIR = "data"

# Z boson mass in GeV (for pairing algorithm)
MZ_PDG = 91.1876  # GeV

# Lepton type codes (ATLAS convention)
ELECTRON = 11
MUON = 13


# ═══════════════════════════════════════════════════════════════
# LORENTZ VECTOR UTILITIES (pure numpy, no external deps)
# ═══════════════════════════════════════════════════════════════

def compute_px_py_pz(pt, eta, phi):
    """Convert (pt, eta, phi) to (px, py, pz)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return px, py, pz


def invariant_mass_from_4vec(e, px, py, pz):
    """Compute invariant mass from (E, px, py, pz). Returns mass in same units."""
    m2 = e**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(m2, 0))  # Protect against numerical noise


def invariant_mass_pair(e1, px1, py1, pz1, e2, px2, py2, pz2):
    """Compute invariant mass of a 2-particle system."""
    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    return invariant_mass_from_4vec(e, px, py, pz)


def invariant_mass_4body(e, px, py, pz):
    """
    Compute 4-body invariant mass.
    e, px, py, pz: arrays of shape (N, 4)
    Returns: array of shape (N,)
    """
    e_tot = np.sum(e, axis=1)
    px_tot = np.sum(px, axis=1)
    py_tot = np.sum(py, axis=1)
    pz_tot = np.sum(pz, axis=1)
    return invariant_mass_from_4vec(e_tot, px_tot, py_tot, pz_tot)


def delta_phi(phi1, phi2):
    """Compute delta phi in [-pi, pi]."""
    dphi = phi1 - phi2
    dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
    return dphi


def delta_r(eta1, phi1, eta2, phi2):
    """Compute delta R = sqrt(deta^2 + dphi^2)."""
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


# ═══════════════════════════════════════════════════════════════
# Z-BOSON PAIRING
# ═══════════════════════════════════════════════════════════════

def pair_z_candidates(e, px, py, pz, charge, lep_type):
    """
    Pair 4 leptons into two Z-boson candidates.

    Algorithm:
      1. Find all valid SFOS (Same Flavour Opposite Sign) pairs.
      2. Among all valid pairings, choose the one where the leading Z
         (Z1) has invariant mass closest to the PDG Z mass (91.19 GeV).
      3. The remaining pair forms Z2.

    For 4 leptons indexed [0,1,2,3], the possible pairings are:
      (0,1)+(2,3), (0,2)+(1,3), (0,3)+(1,2)

    Args:
        e, px, py, pz: shape (N, 4) — 4-vectors of each lepton
        charge: shape (N, 4) — electric charges
        lep_type: shape (N, 4) — lepton type (11=e, 13=mu)

    Returns:
        mZ1, mZ2: shape (N,) — invariant masses of the two Z candidates
        z1_idx, z2_idx: shape (N, 2) — lepton indices forming each Z
        valid_mask: shape (N,) — events where valid pairing was found
    """
    N = e.shape[0]
    mZ1 = np.full(N, np.nan)
    mZ2 = np.full(N, np.nan)
    z1_idx = np.full((N, 2), -1, dtype=int)
    z2_idx = np.full((N, 2), -1, dtype=int)
    valid_mask = np.zeros(N, dtype=bool)

    # All possible ways to split 4 leptons into 2 pairs
    pairings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]

    for event_i in range(N):
        best_dm = 1e12
        found = False

        for (a, b), (c, d) in pairings:
            # Check SFOS for pair (a, b)
            sfos_ab = (charge[event_i, a] + charge[event_i, b] == 0 and
                       lep_type[event_i, a] == lep_type[event_i, b])
            # Check SFOS for pair (c, d)
            sfos_cd = (charge[event_i, c] + charge[event_i, d] == 0 and
                       lep_type[event_i, c] == lep_type[event_i, d])

            if sfos_ab and sfos_cd:
                # Both pairs are valid SFOS
                m_ab = invariant_mass_pair(
                    e[event_i, a], px[event_i, a], py[event_i, a],
                    pz[event_i, a], e[event_i, b], px[event_i, b],
                    py[event_i, b], pz[event_i, b])
                m_cd = invariant_mass_pair(
                    e[event_i, c], px[event_i, c], py[event_i, c],
                    pz[event_i, c], e[event_i, d], px[event_i, d],
                    py[event_i, d], pz[event_i, d])

                # Z1 is the pair closest to MZ, Z2 is the other
                dm_ab = abs(m_ab - MZ_PDG)
                dm_cd = abs(m_cd - MZ_PDG)

                if dm_ab <= dm_cd:
                    z1_m, z2_m = m_ab, m_cd
                    z1_i, z2_i = (a, b), (c, d)
                    dm = dm_ab
                else:
                    z1_m, z2_m = m_cd, m_ab
                    z1_i, z2_i = (c, d), (a, b)
                    dm = dm_cd

                if dm < best_dm:
                    best_dm = dm
                    mZ1[event_i] = z1_m
                    mZ2[event_i] = z2_m
                    z1_idx[event_i] = z1_i
                    z2_idx[event_i] = z2_i
                    found = True

        valid_mask[event_i] = found

    return mZ1, mZ2, z1_idx, z2_idx, valid_mask


def pair_z_candidates_vectorized(e, px, py, pz, charge, lep_type):
    """
    Vectorized version of Z-boson pairing. Much faster for large arrays.
    Same logic as pair_z_candidates but operates on all events at once.
    """
    N = e.shape[0]

    # All possible ways to split 4 leptons into 2 pairs
    pairings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]

    best_mZ1 = np.full(N, np.nan)
    best_mZ2 = np.full(N, np.nan)
    best_z1_idx = np.full((N, 2), -1, dtype=int)
    best_z2_idx = np.full((N, 2), -1, dtype=int)
    best_dm = np.full(N, 1e12)
    valid_mask = np.zeros(N, dtype=bool)

    for (a, b), (c, d) in pairings:
        # Check SFOS for both pairs (vectorized)
        sfos_ab = ((charge[:, a] + charge[:, b] == 0) &
                   (lep_type[:, a] == lep_type[:, b]))
        sfos_cd = ((charge[:, c] + charge[:, d] == 0) &
                   (lep_type[:, c] == lep_type[:, d]))
        both_valid = sfos_ab & sfos_cd

        if not np.any(both_valid):
            continue

        # Compute invariant masses for both pairs
        m_ab = invariant_mass_pair(
            e[:, a], px[:, a], py[:, a], pz[:, a],
            e[:, b], px[:, b], py[:, b], pz[:, b])
        m_cd = invariant_mass_pair(
            e[:, c], px[:, c], py[:, c], pz[:, c],
            e[:, d], px[:, d], py[:, d], pz[:, d])

        # Determine which pair is closer to MZ
        dm_ab = np.abs(m_ab - MZ_PDG)
        dm_cd = np.abs(m_cd - MZ_PDG)

        # Case 1: ab is Z1 (closer to MZ)
        case1 = both_valid & (dm_ab <= dm_cd) & (dm_ab < best_dm)
        best_mZ1[case1] = m_ab[case1]
        best_mZ2[case1] = m_cd[case1]
        best_z1_idx[case1] = [a, b]
        best_z2_idx[case1] = [c, d]
        best_dm[case1] = dm_ab[case1]
        valid_mask[case1] = True

        # Case 2: cd is Z1 (closer to MZ)
        case2 = both_valid & (dm_cd < dm_ab) & (dm_cd < best_dm)
        best_mZ1[case2] = m_cd[case2]
        best_mZ2[case2] = m_ab[case2]
        best_z1_idx[case2] = [c, d]
        best_z2_idx[case2] = [a, b]
        best_dm[case2] = dm_cd[case2]
        valid_mask[case2] = True

    return best_mZ1, best_mZ2, best_z1_idx, best_z2_idx, valid_mask


# ═══════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_features(data, is_mc=True):
    """
    Compute physics-informed features from raw lepton data.

    Args:
        data: dict of numpy arrays loaded from .npz
        is_mc: True for MC, False for real data

    Returns:
        features: dict of numpy arrays (feature_name → values)
        weights: per-event weights (MC) or ones (data)
        valid_mask: events that pass the Z-pairing requirement
    """
    print("  Computing 4-vectors...")
    pt = data["lep_pt"]       # (N, 4) in MeV
    eta = data["lep_eta"]     # (N, 4)
    phi = data["lep_phi"]     # (N, 4)
    e = data["lep_e"]         # (N, 4) in MeV
    charge = data["lep_charge"]  # (N, 4)
    lep_type = data["lep_type"]  # (N, 4)

    N = pt.shape[0]

    # Convert to Cartesian
    px, py, pz = compute_px_py_pz(pt, eta, phi)

    # ─── 1. Four-lepton invariant mass ───
    print("  Computing m4l...")
    m4l = invariant_mass_4body(e, px, py, pz)

    # ─── 2. Z-boson pairing ───
    print("  Pairing Z candidates (vectorized)...")
    mZ1, mZ2, z1_idx, z2_idx, valid_mask = pair_z_candidates_vectorized(
        e, px, py, pz, charge, lep_type)

    n_valid = np.sum(valid_mask)
    n_invalid = N - n_valid
    print(f"  Valid Z-pairings: {n_valid:,} / {N:,} "
          f"({n_invalid:,} events have no valid SFOS pairing)")

    # ─── 3. Additional kinematic features ───
    print("  Computing additional features...")

    # Sort leptons by pT (descending) per event
    pt_sort_idx = np.argsort(-pt, axis=1)  # (N, 4)
    pt_sorted = np.take_along_axis(pt, pt_sort_idx, axis=1)

    features = {
        # ── Mass variables (already in GeV) ──
        "m4l": m4l,
        "mZ1": mZ1,
        "mZ2": mZ2,
        "mZ1_minus_mZ": mZ1 - MZ_PDG,  # Distance from Z mass
        "mZ2_minus_mZ": mZ2 - MZ_PDG,

        # ── pT features (already in GeV) ──
        "lep_pt_0": pt_sorted[:, 0],  # Leading lepton pT
        "lep_pt_1": pt_sorted[:, 1],  # Sub-leading
        "lep_pt_2": pt_sorted[:, 2],  # Third
        "lep_pt_3": pt_sorted[:, 3],  # Softest

        # ── 4-lepton system kinematics ──
        "pt_4l": np.sqrt(np.sum(px, axis=1)**2 +
                         np.sum(py, axis=1)**2),
        "eta_4l": np.arctanh(np.clip(np.sum(pz, axis=1) /
                             (np.sqrt(np.sum(px, axis=1)**2 +
                                      np.sum(py, axis=1)**2 +
                                      np.sum(pz, axis=1)**2) + 1e-10),
                             -0.9999, 0.9999)),

        # ── Lepton eta features ──
        "lep_eta_max": np.max(np.abs(eta), axis=1),
        "lep_eta_min": np.min(np.abs(eta), axis=1),

        # ── Sum of pT ──
        "sum_pt": np.sum(pt, axis=1),

        # ── MET ──
        "met": data["met"] if "met" in data else np.zeros(N),
    }

    # ─── 4. Angular features (require valid Z pairing) ───
    # Only compute for events with valid pairing
    v = valid_mask

    # Delta phi between Z1 and Z2 systems
    dphi_Z1Z2 = np.full(N, 0.0)
    dr_Z1Z2 = np.full(N, 0.0)
    cos_theta_star = np.full(N, 0.0)

    if np.any(v):
        # Z1 and Z2 4-vectors
        z1_a = z1_idx[v, 0]
        z1_b = z1_idx[v, 1]
        z2_a = z2_idx[v, 0]
        z2_b = z2_idx[v, 1]

        # Build index arrays for advanced indexing
        ev_idx = np.arange(N)[v]

        z1_px = px[ev_idx, z1_a] + px[ev_idx, z1_b]
        z1_py = py[ev_idx, z1_a] + py[ev_idx, z1_b]
        z1_pz = pz[ev_idx, z1_a] + pz[ev_idx, z1_b]
        z1_e = e[ev_idx, z1_a] + e[ev_idx, z1_b]

        z2_px = px[ev_idx, z2_a] + px[ev_idx, z2_b]
        z2_py = py[ev_idx, z2_a] + py[ev_idx, z2_b]
        z2_pz = pz[ev_idx, z2_a] + pz[ev_idx, z2_b]
        z2_e = e[ev_idx, z2_a] + e[ev_idx, z2_b]

        # Z system phi and eta
        z1_phi = np.arctan2(z1_py, z1_px)
        z2_phi = np.arctan2(z2_py, z2_px)
        z1_pt_val = np.sqrt(z1_px**2 + z1_py**2)
        z2_pt_val = np.sqrt(z2_px**2 + z2_py**2)
        z1_eta_val = np.arctanh(z1_pz / (np.sqrt(z1_px**2 + z1_py**2 +
                                                   z1_pz**2) + 1e-10))
        z2_eta_val = np.arctanh(z2_pz / (np.sqrt(z2_px**2 + z2_py**2 +
                                                   z2_pz**2) + 1e-10))

        dphi_Z1Z2[v] = delta_phi(z1_phi, z2_phi)
        dr_Z1Z2[v] = delta_r(z1_eta_val, z1_phi, z2_eta_val, z2_phi)

        # cos(theta*) — angle of Z1 relative to beam in 4l rest frame
        # Approximation: cos(theta*) = tanh(Δη(Z1,Z2)/2)
        deta = z1_eta_val - z2_eta_val
        cos_theta_star[v] = np.tanh(deta / 2.0)

    features["dphi_Z1Z2"] = dphi_Z1Z2
    features["dR_Z1Z2"] = dr_Z1Z2
    features["cos_theta_star"] = cos_theta_star

    # ─── 5. Lepton flavor composition ───
    n_electrons = np.sum(np.abs(lep_type) == ELECTRON, axis=1)
    n_muons = np.sum(np.abs(lep_type) == MUON, axis=1)
    # Encode channel: 0=4e, 1=4mu, 2=2e2mu
    channel = np.where(n_electrons == 4, 0,
              np.where(n_muons == 4, 1, 2))
    features["channel"] = channel

    # ─── 6. Isolation features ───
    if "lep_ptvarcone30" in data:
        iso = data["lep_ptvarcone30"] / (pt + 1e-10)
        features["max_iso_ptvarcone30"] = np.max(iso, axis=1)
    if "lep_topoetcone20" in data:
        iso_et = data["lep_topoetcone20"] / (pt + 1e-10)
        features["max_iso_topoetcone20"] = np.max(iso_et, axis=1)

    # ─── 7. Impact parameter ───
    if "lep_d0sig" in data:
        features["max_d0sig"] = np.max(np.abs(data["lep_d0sig"]), axis=1)

    # ─── 8. MC event weights ───
    if is_mc:
        w = data.get("mcWeight", np.ones(N))
        sf_pileup = data.get("ScaleFactor_PILEUP", np.ones(N))
        sf_ele = data.get("ScaleFactor_ELE", np.ones(N))
        sf_muon = data.get("ScaleFactor_MUON", np.ones(N))
        sf_trig = data.get("ScaleFactor_LepTRIGGER", np.ones(N))
        weights = w * sf_pileup * sf_ele * sf_muon * sf_trig
    else:
        weights = np.ones(N)

    return features, weights, valid_mask


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Higgs Boson Classifier -- Feature Engineering")
    print("=" * 60)

    # ─── Load raw data ───
    print("\nLoading raw data...")

    signal_data = dict(np.load(os.path.join(DATA_DIR, "signal_mc.npz")))
    print(f"  Signal MC: {signal_data['lep_pt'].shape[0]:,} events")

    bkg_data = dict(np.load(os.path.join(DATA_DIR, "background_mc.npz")))
    print(f"  Background MC: {bkg_data['lep_pt'].shape[0]:,} events")

    real_data = dict(np.load(os.path.join(DATA_DIR, "collision_data.npz")))
    print(f"  Real data: {real_data['lep_pt'].shape[0]:,} events")

    # ─── Compute features ───
    print("\n--- Signal MC Features ---")
    sig_features, sig_weights, sig_valid = compute_features(signal_data,
                                                             is_mc=True)

    print("\n--- Background MC Features ---")
    bkg_features, bkg_weights, bkg_valid = compute_features(bkg_data,
                                                              is_mc=True)

    print("\n--- Real Data Features ---")
    data_features, data_weights, data_valid = compute_features(real_data,
                                                                 is_mc=False)

    # ─── Apply valid pairing mask ───
    print("\nApplying Z-pairing validity mask...")

    # Get the list of feature names
    feature_names = sorted(sig_features.keys())
    print(f"  Feature count: {len(feature_names)}")
    print(f"  Features: {feature_names}")

    # Build training arrays
    def build_array(features, valid_mask, feature_names):
        n_valid = np.sum(valid_mask)
        X = np.column_stack([features[f][valid_mask] for f in feature_names])
        return X

    X_sig = build_array(sig_features, sig_valid, feature_names)
    X_bkg = build_array(bkg_features, bkg_valid, feature_names)
    X_data = build_array(data_features, data_valid, feature_names)

    w_sig = sig_weights[sig_valid]
    w_bkg = bkg_weights[bkg_valid]
    w_data = data_weights[data_valid]

    # Labels
    y_sig = np.ones(X_sig.shape[0], dtype=np.int32)
    y_bkg = np.zeros(X_bkg.shape[0], dtype=np.int32)

    # Combine MC for training
    X_train = np.vstack([X_sig, X_bkg])
    y_train = np.concatenate([y_sig, y_bkg])
    w_train = np.concatenate([w_sig, w_bkg])

    # Also keep m4l separately for real data (for plotting the Higgs peak)
    m4l_data = data_features["m4l"][data_valid]

    print(f"\n  Signal (valid pairing):     {X_sig.shape[0]:>8,}")
    print(f"  Background (valid pairing): {X_bkg.shape[0]:>8,}")
    print(f"  Training total:             {X_train.shape[0]:>8,}")
    print(f"  Real data (valid pairing):  {X_data.shape[0]:>8,}")

    # ─── Save ───
    print("\nSaving engineered features...")

    np.savez(os.path.join(OUTPUT_DIR, "train_features.npz"),
             X=X_train, y=y_train, weights=w_train,
             feature_names=np.array(feature_names))
    print(f"  Saved train_features.npz")

    np.savez(os.path.join(OUTPUT_DIR, "data_features.npz"),
             X=X_data, weights=w_data, m4l=m4l_data,
             feature_names=np.array(feature_names))
    print(f"  Saved data_features.npz")

    # ─── Quick sanity check ───
    print("\n" + "=" * 60)
    print("SANITY CHECK: m4l Distribution")
    print("=" * 60)

    sig_m4l = sig_features["m4l"][sig_valid]
    bkg_m4l = bkg_features["m4l"][bkg_valid]

    print(f"  Signal m4l:  mean={np.mean(sig_m4l):.1f} GeV, "
          f"median={np.median(sig_m4l):.1f} GeV, "
          f"std={np.std(sig_m4l):.1f} GeV")
    print(f"  Bkg m4l:     mean={np.mean(bkg_m4l):.1f} GeV, "
          f"median={np.median(bkg_m4l):.1f} GeV, "
          f"std={np.std(bkg_m4l):.1f} GeV")
    print(f"  Data m4l:    mean={np.mean(m4l_data):.1f} GeV, "
          f"median={np.median(m4l_data):.1f} GeV, "
          f"std={np.std(m4l_data):.1f} GeV")

    # Count events in Higgs window (120-130 GeV)
    higgs_window = (m4l_data > 120) & (m4l_data < 130)
    print(f"\n  Data events in 120-130 GeV window: {np.sum(higgs_window):,}")
    print(f"  (This is where the Higgs peak should appear)")

    print("\n  Next step: Run 03_train_model.py")


if __name__ == "__main__":
    main()
