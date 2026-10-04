#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drugon — Phase-Encoded Morgan Fingerprints for Affinity Prediction

Author: (your name)
License: MIT

Usage:
    from drugon import Drugon

    model = Drugon()
    model.fit(smiles, pic50)
    y_pred = model.predict(new_smiles)
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
import lightgbm as lgb
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")


class Drugon:
    """
    Phase-Encoded Morgan Fingerprint + LightGBM regressor.

    Parameters
    ----------
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=1024
        Number of bits (before phase encoding).
    lgb_params : dict, optional
        LightGBM hyperparameters. Uses sensible defaults if None.
    """

    def __init__(self, radius=2, n_bits=1024, lgb_params=None):
        self.radius = radius
        self.n_bits = n_bits
        self.lgb_params = lgb_params or {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 7,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.model = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _phase_fingerprint(self, smiles):
        """Compute a 2*n_bits phase-encoded Morgan fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2 * self.n_bits, dtype=np.float32)

        info = {}
        rdMolDescriptors.GetMorganFingerprint(
            mol, self.radius, bitInfo=info
        )

        arr = np.zeros(self.n_bits, dtype=np.complex64)
        for bit, entries in info.items():
            idx = bit % self.n_bits
            for atom_idx, _ in entries:
                atom = mol.GetAtomWithIdx(atom_idx)
                phase = (
                    atom.GetAtomicNum() * 0.15
                    + (1.0 if atom.GetIsAromatic() else 0.0) * 1.5
                    + atom.GetFormalCharge() * 1.2
                ) * np.pi
                arr[idx] += np.exp(1j * phase)

        return np.concatenate([arr.real, arr.imag]).astype(np.float32)

    def _transform(self, smiles_list):
        """Transform a list of SMILES to feature matrix."""
        return np.array([self._phase_fingerprint(s) for s in smiles_list])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, smiles_list, y):
        """Train Drugon on SMILES and target values (pIC50)."""
        X = self._transform(smiles_list)
        self.model = lgb.LGBMRegressor(**self.lgb_params)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, smiles_list):
        """Predict pIC50 for new SMILES."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X = self._transform(smiles_list)
        return self.model.predict(X)

    def feature_importance(self, top_k=20):
        """Return top-k most important features."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        imp = self.model.feature_importances_
        return np.argsort(imp)[::-1][:top_k]

    # ------------------------------------------------------------------
    # Evaluation utilities
    # ------------------------------------------------------------------
    def cross_validate(self, smiles_list, y, n_splits=5, random_state=42):
        """Run k-fold CV and return (r, rho, rmse)."""
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import mean_squared_error

        X = self._transform(smiles_list)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof = np.zeros(len(y))

        for tr, va in kf.split(X):
            m = lgb.LGBMRegressor(**self.lgb_params)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])

        r = pearsonr(y, oof)[0]
        rho = spearmanr(y, oof)[0]
        rmse = np.sqrt(mean_squared_error(y, oof))
        return r, rho, rmse


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import pandas as pd

    if len(sys.argv) < 2:
        print("Usage: python drugon.py <chembl_file.smi>")
        sys.exit(1)

    path = sys.argv[1]
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 3:
                continue
            smi, _, val = parts[0], parts[1], parts[2]
            try:
                val = float(val)
            except ValueError:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if 15 <= mol.GetNumAtoms() <= 50:
                rows.append((smi, val))

    print(f"Loaded {len(rows)} molecules from {path}")
    smiles = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])

    model = Drugon()
    r, rho, rmse = model.cross_validate(smiles, y)
    print(f"\nDrugon 5-fold CV:")
    print(f"  Pearson  r:  {r:+.3f}")
    print(f"  Spearman ρ:  {rho:+.3f}")
    print(f"  RMSE:        {rmse:.3f}")
