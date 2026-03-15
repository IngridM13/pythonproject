"""
Noise injection module for HDC recall experiments.

Provides inject_noise() to corrupt a normalized person dict in a controlled,
reproducible way. No pytest dependencies.

Corruption strategies are designed to reflect realistic data quality issues
encountered when reconciling records from different sources (data entry errors,
life changes, different capture moments, encoding inconsistencies).
"""

import random
import copy
import unicodedata
from datetime import timedelta, date as date_cls

# Valid categorical values (must match what the encoder expects)
_MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"]
_GENDERS = ["Male", "Female", "Non-binary", "Other"]
_RACES = ["White", "Black", "Asian", "Hispanic", "Mixed", "Other"]


# ---------------------------------------------------------------------------
# String corruption helpers
# ---------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    """Normalize accented characters to their ASCII equivalents (e.g. María → Maria)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _realistic_name_typo(s: str, rng: random.Random) -> str:
    """
    Apply one realistic string corruption to a name:
      - transposition: swap two adjacent characters  (e.g. John  → Jonh)
      - deletion:      remove one character           (e.g. John  → Jon)
      - insertion:     insert a random letter         (e.g. John  → Johnn)
      - substitution:  replace one character          (e.g. John  → Jahn)
      - accent strip:  remove diacritics              (e.g. María → Maria)
    """
    if not s:
        return s

    ops = ["transposition", "deletion", "insertion", "substitution", "accent_strip"]
    op = rng.choice(ops)

    if op == "accent_strip":
        result = _strip_accents(s)
        if result != s:
            return result
        # String has no accents — fall back to substitution
        op = "substitution"

    chars = list(s)

    if op == "transposition" and len(chars) >= 2:
        idx = rng.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    elif op == "deletion" and len(chars) >= 2:
        idx = rng.randint(0, len(chars) - 1)
        chars.pop(idx)

    elif op == "insertion":
        idx = rng.randint(0, len(chars))
        letter = rng.choice("abcdefghijklmnopqrstuvwxyz")
        if idx < len(chars) and chars[idx].isupper():
            letter = letter.upper()
        chars.insert(idx, letter)

    else:  # substitution
        if chars:
            idx = rng.randint(0, len(chars) - 1)
            original = chars[idx].lower()
            candidates = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != original]
            replacement = rng.choice(candidates)
            if chars[idx].isupper():
                replacement = replacement.upper()
            chars[idx] = replacement

    return "".join(chars)


# ---------------------------------------------------------------------------
# Date corruption helpers
# ---------------------------------------------------------------------------

def _perturb_date(d, rng: random.Random):
    """
    Apply one realistic date corruption:
      - day/month swap:  swap day and month values if the result is a valid date
                         (reflects common data entry transposition, e.g. 15/05 → 05/15)
      - year offset:     shift year by ±1–5 years
                         (reflects age discrepancy between sources)
      - day offset:      shift by ±1–30 days
                         (minor transcription error)
    """
    if d is None:
        return d

    op = rng.choice(["day_month_swap", "year_offset", "day_offset"])

    if op == "day_month_swap":
        try:
            swapped = date_cls(d.year, d.day, d.month)
            if swapped != d:
                return swapped
        except ValueError:
            pass  # day > 12, swap not valid — fall through to day_offset
        op = "day_offset"

    if op == "year_offset":
        offset = rng.randint(1, 5) * rng.choice([-1, 1])
        try:
            return date_cls(d.year + offset, d.month, d.day)
        except ValueError:
            return d

    # day_offset
    sign = rng.choice([-1, 1])
    delta = timedelta(days=rng.randint(1, 30)) * sign
    return d + delta


# ---------------------------------------------------------------------------
# Phone number corruption helpers
# ---------------------------------------------------------------------------

def _perturb_phone(s: str, rng: random.Random) -> str:
    """
    Apply one realistic phone corruption:
      - digit error (65%): change 1–3 digits (transcription error)
      - new number (35%):  randomize all digits (person got a new phone number,
                           reflects records captured at different moments in life)
    """
    if not s:
        return s

    chars = list(s)
    digit_positions = [i for i, c in enumerate(chars) if c.isdigit()]
    if not digit_positions:
        return s

    if rng.random() < 0.65:
        # Change 1–3 digits
        n_changes = min(rng.randint(1, 3), len(digit_positions))
        for idx in rng.sample(digit_positions, n_changes):
            original = chars[idx]
            candidates = [str(d) for d in range(10) if str(d) != original]
            chars[idx] = rng.choice(candidates)
    else:
        # Replace all digits (new phone number)
        for i in digit_positions:
            chars[i] = str(rng.randint(0, 9))

    return "".join(chars)


# ---------------------------------------------------------------------------
# List field corruption helpers
# ---------------------------------------------------------------------------

def _perturb_address_list(addresses: list, rng: random.Random) -> list:
    """
    Realistic address list corruption reflecting life changes:
      - remove one address   (person moved; old address missing from new source)
      - modify street number (transcription error in one digit)

    The list is never fully cleared — partial overlap is preserved to reflect
    that the same person may share some addresses across sources.
    """
    if not addresses:
        return addresses

    addresses = list(addresses)
    op = rng.choice(["remove_one", "modify_number"])

    if op == "remove_one" and len(addresses) > 1:
        addresses.pop(rng.randint(0, len(addresses) - 1))

    else:  # modify_number
        idx = rng.randint(0, len(addresses) - 1)
        addr = addresses[idx]
        digit_positions = [i for i, c in enumerate(addr) if c.isdigit()]
        if digit_positions:
            chars = list(addr)
            pos = rng.choice(digit_positions)
            candidates = [str(d) for d in range(10) if str(d) != chars[pos]]
            chars[pos] = rng.choice(candidates)
            addresses[idx] = "".join(chars)

    return addresses


def _perturb_string_list(values: list, rng: random.Random) -> list:
    """
    Realistic corruption for akas and landlines:
      - remove one element  (not recorded in this source)
      - modify one element  (transcription error — digit change for phones,
                             name typo for akas)

    The list is never fully cleared.
    """
    if not values:
        return values

    values = list(values)
    op = rng.choice(["remove_one", "modify_one"])

    if op == "remove_one" and len(values) > 1:
        values.pop(rng.randint(0, len(values) - 1))

    else:  # modify_one
        idx = rng.randint(0, len(values) - 1)
        v = values[idx]
        if any(c.isdigit() for c in v):
            values[idx] = _perturb_phone(v, rng)
        else:
            values[idx] = _realistic_name_typo(v, rng)

    return values


# ---------------------------------------------------------------------------
# Categorical helpers
# ---------------------------------------------------------------------------

def _different_category(current: str, options: list, rng: random.Random) -> str:
    """Return a random value from options that differs from current."""
    current_cap = current.strip().capitalize() if current else ""
    candidates = [o for o in options if o.capitalize() != current_cap]
    if not candidates:
        candidates = options
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Field corruption dispatch table
# ---------------------------------------------------------------------------

_NOISE_FUNCS = {
    "name":            lambda v, rng: _realistic_name_typo(v, rng),
    "lastname":        lambda v, rng: _realistic_name_typo(v, rng),
    "dob":             lambda v, rng: _perturb_date(v, rng),
    "marital_status":  lambda v, rng: _different_category(v, _MARITAL_STATUSES, rng),
    "mobile_number":   lambda v, rng: _perturb_phone(v, rng),
    "gender":          lambda v, rng: _different_category(v, _GENDERS, rng),
    "race":            lambda v, rng: _different_category(v, _RACES, rng),
    "attrs.address":   lambda v, rng: _perturb_address_list(v, rng),
    "attrs.akas":      lambda v, rng: _perturb_string_list(v, rng),
    "attrs.landlines": lambda v, rng: _perturb_string_list(v, rng),
}

_CORRUPTIBLE_FIELDS = list(_NOISE_FUNCS.keys())  # fixed order for reproducibility


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    Notes
    -----
    Corruption strategies reflect realistic data quality issues:
    - Name/lastname: transposition, deletion, insertion, substitution, accent stripping
    - DOB: day/month swap, year offset (±1–5 yr), or small day shift (±1–30 d)
    - Mobile: digit error (1–3 digits) or completely new number (35% chance)
    - Address: remove one entry or modify a street number (never fully cleared)
    - Akas/landlines: remove one entry or modify one (never fully cleared)
    - Categorical fields: replaced with a different valid category
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
