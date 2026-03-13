"""
Near-duplicate generation for HDC recall experiments.

Provides generate_near_duplicates() to create confuser records that share
key fields with real persons in the collection, making the search problem
harder and more realistic.

A near-duplicate is a *different* person that shares key fields with an
existing record. Their presence forces the HDC search to distinguish between
structurally similar but distinct identities.

No pytest dependencies — usable from any context.
"""

import copy
import random

from tests.experiments.noise_injection import (
    _realistic_name_typo,
    _perturb_date,
    _different_category,
    _MARITAL_STATUSES,
    _GENDERS,
    _RACES,
)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _near_dup_same_name(source: dict, rng: random.Random) -> dict:
    """
    Shares name + lastname with source; DOB is perturbed.

    Represents a common real-world scenario: two people with the same name
    born on a slightly different date (data entry error or actual different
    person with the same name).
    """
    dup = copy.deepcopy(source)
    dup["dob"] = _perturb_date(dup.get("dob"), rng)
    return dup


def _near_dup_same_dob(source: dict, rng: random.Random) -> dict:
    """
    Shares DOB with source; name receives a realistic typo.

    Represents records where the date of birth matches exactly but the name
    was transcribed with a minor error.
    """
    dup = copy.deepcopy(source)
    dup["name"] = _realistic_name_typo(dup.get("name", ""), rng)
    return dup


def _near_dup_same_name_dob(source: dict, rng: random.Random) -> dict:
    """
    Shares name + lastname + DOB with source; one categorical field is changed.

    Maximum confuser: only marital_status, gender, or race differs. This
    tests whether the encoder can distinguish records that are nearly identical
    in all searchable string/date fields.
    """
    dup = copy.deepcopy(source)
    field = rng.choice(["marital_status", "gender", "race"])
    if field == "marital_status":
        dup["marital_status"] = _different_category(
            dup.get("marital_status", ""), _MARITAL_STATUSES, rng
        )
    elif field == "gender":
        dup["gender"] = _different_category(
            dup.get("gender", ""), _GENDERS, rng
        )
    else:
        dup["race"] = _different_category(
            dup.get("race", ""), _RACES, rng
        )
    return dup


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "same_name":     _near_dup_same_name,
    "same_dob":      _near_dup_same_dob,
    "same_name_dob": _near_dup_same_name_dob,
}

_STRATEGY_NAMES = list(_STRATEGIES.keys())  # fixed order for reproducibility


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_near_duplicates(
    persons: list,
    n: int,
    rng: random.Random,
) -> list:
    """
    Return a list of `n` near-duplicate person dicts drawn from `persons`.

    Each near-duplicate is created by:
    1. Randomly selecting a source record from `persons`.
    2. Randomly selecting one of three strategies (uniform distribution).
    3. Applying the strategy to produce a mutated deep copy.

    Strategies
    ----------
    same_name     : copy source, perturb DOB only (shares name + lastname)
    same_dob      : copy source, apply one name typo (shares DOB)
    same_name_dob : copy source, change one categorical field
                    (marital_status, gender, or race) — maximum confuser

    Parameters
    ----------
    persons : list of dict
        Normalized person dicts (output of normalize_person_data).
        Must be non-empty.
    n : int
        Number of near-duplicates to generate. May be zero (returns []).
    rng : random.Random
        Seeded RNG for full reproducibility.

    Returns
    -------
    list of dict
        `n` near-duplicate person dicts. These are deep copies — the caller's
        `persons` list is never mutated. The caller is responsible for
        inserting them into the collection; they should NOT be added to
        id_to_person (they are confusers, not query targets).

    Raises
    ------
    ValueError
        If `persons` is empty and `n > 0`.
    """
    if n == 0:
        return []
    if not persons:
        raise ValueError(
            "generate_near_duplicates: `persons` must be non-empty when n > 0"
        )

    result = []
    for _ in range(n):
        source = rng.choice(persons)
        strategy_name = rng.choice(_STRATEGY_NAMES)
        strategy_fn = _STRATEGIES[strategy_name]
        result.append(strategy_fn(source, rng))

    return result
