"""Human-readable summary of a TwoNN / PCA-95 intrinsic-dimension profile."""

from __future__ import annotations

from typing import Dict, List, Mapping


def interpret_id_profile(id_profile: Mapping[str, float]) -> Dict[str, object]:
    """Summarise TwoNN and PCA-95 for the report."""
    pca95 = float(id_profile.get("intrinsic_dim_pca_95") or 0.0)
    twonn = float(id_profile.get("intrinsic_dim_twonn") or 0.0)

    narrative: List[str] = [
        f"PCA-95 ID ≈ {pca95:.0f}, TwoNN ≈ {twonn:.1f}."
    ]
    if pca95 > 0 and twonn > 0 and abs(pca95 - twonn) / max(pca95, 1.0) > 0.5:
        narrative.append(
            "PCA and TwoNN disagree substantially — the embedding geometry is likely nonlinear."
        )
    return {
        "framework": "TwoNN / PCA-95",
        "narrative": narrative,
        "estimators": {
            "pca_95": round(pca95, 4),
            "twonn": round(twonn, 4),
        },
    }
