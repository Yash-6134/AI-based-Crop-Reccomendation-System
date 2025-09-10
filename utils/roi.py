# utils/roi.py
from typing import Optional, Dict
from flask_babel import gettext as _  # ready for any future user-facing messages [no strings yet]

# MSP crops in ₹ per quintal (update every season from official MSP notices)
# NOTE: Keep keys un-translated for stable lookups; UI shows translated labels elsewhere.
DEFAULT_MSP_RSQ = {
    "rice": 2300,
    "maize": 2225,
    "cotton": 7121,   # seed-cotton/kapas
    "jute": 5650,
    "chickpea": 5440
}

# Rough default yields (edit per district if you have better priors)
# NOTE: Keep keys un-translated for stable lookups; localize labels in templates.
DEFAULT_YIELD_Q_PER_ACRE = {"rice": 10, "maize": 14, "cotton": 8, "jute": 8, "chickpea": 6}
DEFAULT_YIELD_KG_PER_ACRE = {
    "banana": 30000, "mango": 4000, "grapes": 8000,
    "orange": 5000, "apple": 4000, "coffee": 800, "watermelon": 10000
}

MSP_CROPS = set(DEFAULT_MSP_RSQ.keys())

def is_msp_crop(crop: str) -> bool:
    return crop.lower() in MSP_CROPS

def compute_roi(
    crop: str,
    price_per_quintal: Optional[float] = None,
    price_per_kg: Optional[float] = None,
    yield_q_per_acre: Optional[float] = None,
    yield_kg_per_acre: Optional[float] = None,
    total_cost_per_acre: float = 0.0,
) -> Dict[str, float]:
    c = crop.lower()

    if is_msp_crop(c) or price_per_quintal:
        p = price_per_quintal if price_per_quintal is not None else DEFAULT_MSP_RSQ.get(c, 0.0)
        y = yield_q_per_acre if yield_q_per_acre is not None else DEFAULT_YIELD_Q_PER_ACRE.get(c, 0.0)
        gross = p * y  # ₹
    else:
        p = price_per_kg if price_per_kg is not None else 0.0
        y = yield_kg_per_acre if yield_kg_per_acre is not None else DEFAULT_YIELD_KG_PER_ACRE.get(c, 0.0)
        gross = p * y  # ₹

    net = gross - (total_cost_per_acre or 0.0)
    # Keys 'gross' and 'net' are programmatic; labels are translated in templates (result.html).
    return {"gross": round(gross, 2), "net": round(net, 2)}
