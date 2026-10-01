def classify_period(p_bls: float, p_ref: float, tol: float = 0.01) -> str:
    """hit within tol; harmonic at 0.5x, 2x, 1/3x, 3x; otherwise miss."""
    ratio = p_bls / p_ref
    if abs(ratio - 1) < tol:
        return "hit"
    for h in (0.5, 2.0, 1 / 3, 3.0):
        if abs(ratio / h - 1) < tol:
            return "harmonic"
    return "miss"
