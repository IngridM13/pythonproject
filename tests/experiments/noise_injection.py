"""
Noise injection module for HDC recall experiments.

Provides inject_noise() to corrupt a normalized person dict in a controlled,
reproducible way. No pytest dependencies.
"""

import random
import copy
from datetime import timedelta

# Valid categorical values (must match what the encoder expects)
_MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"]
_GENDERS = ["Male", "Female", "Non-binary", "Other"]
_RACES = ["White", "Black", "Asian", "Hispanic", "Mixed", "Other"]


def _typo(s: str, rng: random.Random) -> str:
    """Replace one random character in s with a different ASCII letter."""
    if not s:
        return s
    idx = rng.randint(0, len(s) - 1)
    chars = list(s)
    original = chars[idx].lower()
    # Pick a different letter
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    candidates = [c for c in alphabet if c != original]
    replacement = rng.choice(candidates)
    # Preserve original case
    if chars[idx].isupper():
        replacement = replacement.upper()
    chars[idx] = replacement
    return "".join(chars)


def _typo_digit(s: str, rng: random.Random) -> str:
    """Replace one random digit in s with a different digit."""
    if not s:
        return s
    digit_positions = [i for i, c in enumerate(s) if c.isdigit()]
    if not digit_positions:
        return s
    idx = rng.choice(digit_positions)
    chars = list(s)
    original = chars[idx]
    candidates = [str(d) for d in range(10) if str(d) != original]
    chars[idx] = rng.choice(candidates)
    return "".join(chars)


def _different_category(current: str, options: list, rng: random.Random) -> str:
    """Return a random value from options that differs from current."""
    # Normalize comparison: capitalize to match normalize_person_data behaviour
    current_cap = current.strip().capitalize() if current else ""
    candidates = [o for o in options if o.capitalize() != current_cap]
    if not candidates:
        candidates = options
    return rng.choice(candidates)


def _perturb_date(d, rng: random.Random):
    """Shift a date by ±1–30 days."""
    from datetime import date as date_cls
    if d is None:
        return d
    delta = timedelta(days=rng.randint(1, 30))
    if rng.random() < 0.5:
        delta = -delta
    new_date = d + delta
    # Keep date within reasonable bounds
    return new_date


# Maps each corruptible field to its noise function.
# Functions receive (value, rng) and return the noisy value.
_NOISE_FUNCS = {
    "name":           lambda v, rng: _typo(v, rng),
    "lastname":       lambda v, rng: _typo(v, rng),
    "dob":            lambda v, rng: _perturb_date(v, rng),
    "marital_status": lambda v, rng: _different_category(v, _MARITAL_STATUSES, rng),
    "mobile_number":  lambda v, rng: _typo_digit(v, rng),
    "gender":         lambda v, rng: _different_category(v, _GENDERS, rng),
    "race":           lambda v, rng: _different_category(v, _RACES, rng),
    "attrs.address":  lambda v, rng: [],
    "attrs.akas":     lambda v, rng: [],
    "attrs.landlines": lambda v, rng: [],
}

_CORRUPTIBLE_FIELDS = list(_NOISE_FUNCS.keys())  # fixed order for reproducibility


def inject_noise(person: dict, noise_fraction: float, rng: random.Random) -> dict:
    """
    Return a copy of `person` with a controlled number of fields corrupted.

    Parameters
    ----------
    person : dict
        Normalized person dict (output of normalize_person_data).
    noise_fraction : float
        Fraction of the 10 corruptible fields to corrupt. 0.0 → no corruption,
        1.0 → all 10 fields corrupted.
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    dict
        Deep copy of person with floor(noise_fraction * 10) fields corrupted.
    """
    import math
    noisy = copy.deepcopy(person)
    n_corrupt = math.floor(noise_fraction * len(_CORRUPTIBLE_FIELDS))
    if n_corrupt == 0:
        return noisy

    fields_to_corrupt = rng.sample(_CORRUPTIBLE_FIELDS, n_corrupt)

    for field in fields_to_corrupt:
        func = _NOISE_FUNCS[field]
        if field.startswith("attrs."):
            attr_key = field.split(".", 1)[1]
            attrs = noisy.get("attrs")
            if isinstance(attrs, dict):
                current = attrs.get(attr_key, [])
                noisy["attrs"][attr_key] = func(current, rng)
        else:
            current = noisy.get(field)
            noisy[field] = func(current, rng)

    return noisy
