from typing import Dict, Any
from datetime import date as date_cls
from dateutil.parser import parse as dateutil_parse

# Copy the necessary constants from encoding_and_search_milvus.py
DEFAULT_SCALARS = {
    "name": "",
    "lastname": "",
    "dob": None,
    "marital_status": "",
    "mobile_number": "",
    "gender": "",
    "race": ""
}

DEFAULT_ATTRS = {
    "address": [],
    "akas": [],
    "landlines": []
}


def _as_list_str(value):
    """Ensure value is a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    # Flatten list-of-lists if needed and convert all elements to strings
    result = []
    for item in value:
        if isinstance(item, list):
            result.extend([str(x) for x in item if x is not None])
        elif item is not None:
            result.append(str(item))
    return result


def _is_iso_date(s):
    """Check if string follows ISO date format YYYY-MM-DD."""
    if not isinstance(s, str):
        return False
    parts = s.split("-")
    if len(parts) != 3:
        return False
    try:
        return all(part.isdigit() for part in parts) and len(parts[0]) == 4
    except (IndexError, AttributeError):
        return False


def parse_date(s: str | None) -> date_cls | None:
    """
    Convert a date string to datetime.date.
    Accepts '' or None -> None.
    Handles various common date formats (ISO, slashes, etc.).
    Raises ValueError for unparseable input.
    """
    if s in (None, ""):
        return None

    # This was the fix for your previous problem (keep it)
    if isinstance(s, date_cls):
        return s

    if not isinstance(s, str):
        raise TypeError("dob must be a string, '', or None")

    try:
        # Use dateutil.parse to flexibly handle formats
        return dateutil_parse(s).date()
    except (ValueError, OverflowError) as e:
        # Re-raise with a clear message
        raise ValueError(f"Could not parse '{s}' as a valid date") from e


def normalize_person_data(person: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(person, dict):
        raise ValueError("person must be a dict")

    out: Dict[str, Any] = {}

    # Inicializa con defaults de escalares
    out.update(DEFAULT_SCALARS)

    # Procesa claves de entrada
    for k, v in person.items():
        lk = k.lower()
        if lk == "attrs":
            continue  # se maneja luego
        if lk == "dob":
            out["dob"] = parse_date(v)
        elif lk in ("name", "lastname", "mobile_number", "race"):
            out[lk] = "" if v in (None, "") else str(v).strip()
        elif lk in ("marital_status", "gender"):
            out[lk] = "" if v in (None, "") else str(v).strip().capitalize()
        else:
            # si aparece otro escalar no-lista, lo guardamos como string
            if not isinstance(v, list):
                out[lk] = "" if v in (None, "") else str(v)

    # Asegura attrs y sus listas conocidas
    attrs_in = person.get("attrs")
    attrs: Dict[str, Any] = dict(attrs_in) if isinstance(attrs_in, dict) else {}

    # Si address vino al top-level, muévelo a attrs (tu test lo manda así)
    if "address" in person and "address" not in attrs:
        attrs["address"] = person.get("address")
    print(f"[normalize_person_data] attrs: {attrs}")
    # Normaliza listas conocidas
    for key, default_list in DEFAULT_ATTRS.items():
        attrs[key] = _as_list_str(attrs.get(key, default_list))

    out["attrs"] = attrs

    return out