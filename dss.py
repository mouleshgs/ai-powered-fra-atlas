"""
Simple Decision Support System (DSS) for FRA applicants.
Provides rule-based eligibility checks and priority scoring.

Functions:
- recommend_schemes(applicant) -> list[str]
- recommend_with_priority(applicant) -> list[dict]

Applicant schema (dict) - expected keys (some optional):
- name: str
- land_size: float (hectares)
- tribe: bool
- water_access: str in {'piped', 'well', 'none'}
- income: float (annual INR)
- age: int
- gender: str
- household_size: int
- agriculture: bool (does household farm?)
- labourer: bool (migratory or daily-wage labour)
- housing_deficit: bool
- disability: bool
- female_headed: bool

This is rule-based and conservative; easily extensible and testable.
"""
from typing import Dict, List, Any
import math


def normalize_applicant(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate applicant dict, expand schema with defaults.

    New/expanded fields added:
    - geolocation: dict with keys {'lat', 'lon'} (floats) or None
    - patta_confidence: float in [0,1] (OCR/confidence of patta fields)
    - scanned_metadata: dict with optional keys like {'dpi', 'scanner', 'timestamp'}

    The function coerces types where possible and ensures required keys exist with defaults.
    """
    a = dict(raw or {})
    # Basic normalizations
    try:
        a['land_size'] = float(a.get('land_size') or 0.0)
    except Exception:
        a['land_size'] = 0.0
    try:
        a['income'] = float(a.get('income') or 0.0)
    except Exception:
        a['income'] = 0.0
    try:
        a['age'] = int(a.get('age') or 0)
    except Exception:
        a['age'] = 0
    try:
        a['household_size'] = int(a.get('household_size') or 1)
    except Exception:
        a['household_size'] = 1

    # Booleans
    for k in ('tribe', 'agriculture', 'labourer', 'housing_deficit', 'disability', 'female_headed'):
        v = a.get(k)
        if isinstance(v, str):
            a[k] = v.strip().lower() in ('1', 'true', 'yes', 'y')
        else:
            a[k] = bool(v)

    # Water access normalization
    wa = (a.get('water_access') or '').strip().lower()
    if wa in ('piped', 'tap', 'municipal'):
        a['water_access'] = 'piped'
    elif wa in ('well', 'borewell', 'tube well', 'tubewell'):
        a['water_access'] = 'well'
    elif wa in ('none', '', None):
        a['water_access'] = 'none'
    else:
        a['water_access'] = wa

    # Geolocation
    geo = a.get('geolocation')
    if isinstance(geo, dict):
        try:
            lat = float(geo.get('lat'))
            lon = float(geo.get('lon'))
            if math.isfinite(lat) and math.isfinite(lon):
                a['geolocation'] = {'lat': lat, 'lon': lon}
            else:
                a['geolocation'] = None
        except Exception:
            a['geolocation'] = None
    else:
        a['geolocation'] = None

    # Patta confidence
    try:
        pc = float(a.get('patta_confidence', 0.0) or 0.0)
        a['patta_confidence'] = max(0.0, min(1.0, pc))
    except Exception:
        a['patta_confidence'] = 0.0

    # Scanned metadata
    md = a.get('scanned_metadata')
    if isinstance(md, dict):
        a['scanned_metadata'] = md
    else:
        a['scanned_metadata'] = {}

    return a

# Define scheme rules as functions that accept applicant dict and return bool

def rule_pm_kisan(applicant: Dict[str, Any]) -> bool:
    """PM-KISAN: income support for small/marginal farmers.
    Eligibility heuristics:
    - applicant must be an agricultural household (agriculture=True)
    - land_size > 0 and <= 2 hectares (small/marginal)
    - not excluded by very high income (> 100000)
    """
    try:
        if not applicant.get('agriculture', False):
            return False
        ls = float(applicant.get('land_size', 0))
        income = float(applicant.get('income', 0))
        if ls <= 0:
            return False
        if ls > 2.0:
            return False
        if income > 100000:
            return False
        return True
    except Exception:
        return False


def rule_jal_jeevan_mission(applicant: Dict[str, Any]) -> bool:
    """Jal Jeevan Mission (piped drinking water to households).
    Heuristic:
    - Eligible if water_access is 'none' or 'well' (no piped connection).
    - Prioritize households with young children or elderly (age>60 or household_size>4)
    """
    wa = applicant.get('water_access', '').lower()
    if wa in ('piped', 'tap', 'municipal'):
        return False
    # If explicit 'none' or 'well' or empty -> eligible
    return True


def rule_mgnrega(applicant: Dict[str, Any]) -> bool:
    """MGNREGA: employment guarantee for rural households.
    Heuristic:
    - Eligible if household is rural agricultural labourer or household income below a threshold
    - Age: working-age members present (between 18 and 60)
    - If labourer=True or income < 120000 -> eligible
    """
    try:
        labour = applicant.get('labourer', False)
        income = float(applicant.get('income', 0))
        age = int(applicant.get('age', 0))
        if labour:
            return True
        if income < 120000:
            return True
        # If age in working range and household size suggests need
        if 18 <= age <= 60 and applicant.get('household_size', 1) >= 3:
            return True
        return False
    except Exception:
        return False


def rule_dajgua(applicant: Dict[str, Any]) -> bool:
    """DAJGUA: (Assumed tribal support scheme) - preferential for Scheduled Tribes or vulnerable tribal households.
    Heuristic:
    - Eligible if applicant['tribe'] is True
    - Or if household is below-poverty (income < 80000) and large household
    """
    try:
        if applicant.get('tribe', False):
            return True
        income = float(applicant.get('income', 1e9))
        if income < 80000 and applicant.get('household_size', 1) >= 5:
            return True
        return False
    except Exception:
        return False


# Map scheme keys to rule functions and metadata
SCHEMES = {
    'PM-KISAN': {
        'rule': rule_pm_kisan,
        'category': 'income_support',
        'priority_weight': 0.8,
        'description': 'Direct income support to small and marginal farmer families to supplement their financial needs.',
        'url': 'https://pmkisan.gov.in/',
    },
    'Jal Jeevan Mission': {
        'rule': rule_jal_jeevan_mission,
        'category': 'water',
        'priority_weight': 1.0,
        'description': 'National programme to provide safe and adequate drinking water through individual household tap connections by 2024 to all rural households.',
        'url': 'https://jaljeevanmission.gov.in/',
    },
    'MGNREGA': {
        'rule': rule_mgnrega,
        'category': 'employment',
        'priority_weight': 0.6,
        'description': 'Mahatma Gandhi National Rural Employment Guarantee Act — guarantees 100 days of wage employment to rural households, focusing on livelihood security.',
        'url': 'https://nrega.nic.in/',
    },
    'DAJGUA': {
        'rule': rule_dajgua,
        'category': 'tribal_support',
        'priority_weight': 0.9,
        'description': 'Assistance and targeted support programmes for Scheduled Tribes and vulnerable tribal households (state/central tribal welfare schemes).',
        'url': 'https://tribal.nic.in/',
    }
}


def recommend_schemes(applicant: Dict[str, Any]) -> List[str]:
    """Return list of eligible scheme names for the applicant."""
    a = normalize_applicant(applicant)
    eligible = []
    for name, meta in SCHEMES.items():
        rule_fn = meta.get('rule')
        try:
            if rule_fn(a):
                eligible.append(name)
        except Exception:
            # If a rule crashes, skip
            continue
    return eligible


def recommend_with_priority(applicant: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of eligible schemes with priority scores and reasons.

    Scoring approach (simple weighted heuristic):
    - Start with scheme priority_weight
    - Add bonus depending on applicant attributes matching scheme category (e.g., water_access for water)
    - Normalize final scores to 0-100 scale
    """
    a = normalize_applicant(applicant)
    scored = []
    base_scores = []
    for name, meta in SCHEMES.items():
        rule_fn = meta.get('rule')
        if not rule_fn(a):
            continue
        weight = float(meta.get('priority_weight', 0.5))
        score = weight * 50  # base
        reasons = [meta.get('description', '')]
        cat = meta.get('category', '')
        # Category-specific boosts
        if cat == 'water':
            wa = a.get('water_access', '').lower()
            if wa in ('none', ''):
                score += 30
                reasons.append('No piped water')
            elif wa in ('well',):
                score += 10
                reasons.append('Well only')
        if cat == 'income_support':
            ls = float(a.get('land_size', 0) or 0)
            income = float(a.get('income', 0) or 0)
            if ls > 0 and ls <= 1.0:
                score += 20
                reasons.append('Small landholding')
            if income < 50000:
                score += 20
                reasons.append('Low income')
        if cat == 'tribal_support':
            if a.get('tribe', False):
                score += 40
                reasons.append('Declared tribal status')
            elif float(a.get('income', 1e9)) < 60000:
                score += 10
                reasons.append('Low income tribal-like')
        if cat == 'employment':
            if a.get('labourer', False):
                score += 25
                reasons.append('Household labourer')
            if 18 <= int(a.get('age', 0)) <= 60:
                score += 5
        base_scores.append(score)
        scored.append({
            'scheme': name,
            'raw_score': score,
            'reasons': reasons,
            'description': meta.get('description', ''),
            'url': meta.get('url', '')
        })
    # Normalize scores to 0-100
    if not scored:
        return []
    min_s = min(s['raw_score'] for s in scored)
    max_s = max(s['raw_score'] for s in scored)
    if max_s == min_s:
        # all same, scale to [50,80] roughly
        for s in scored:
            s['score'] = int(min(100, max(0, s['raw_score'] * 1)))
    else:
        for s in scored:
            s['score'] = int(100 * (s['raw_score'] - min_s) / (max_s - min_s))
    # Sort by score desc
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored


# ML integration stub - feature extractor
def extract_features_for_ml(applicant: Dict[str, Any]) -> Dict[str, float]:
    """Convert applicant dict to numeric features ready for ML models.

    Example features (normalized): land_size, income, household_size, water_flag, agriculture_flag, tribe_flag
    """
    f = {}
    f['land_size'] = float(applicant.get('land_size', 0) or 0)
    f['income'] = float(applicant.get('income', 0) or 0)
    f['household_size'] = float(applicant.get('household_size', 1) or 1)
    wa = applicant.get('water_access', '').lower()
    f['water_flag'] = 1.0 if wa in ('piped', 'tap') else 0.0
    f['agriculture_flag'] = 1.0 if applicant.get('agriculture', False) else 0.0
    f['tribe_flag'] = 1.0 if applicant.get('tribe', False) else 0.0
    f['labourer_flag'] = 1.0 if applicant.get('labourer', False) else 0.0
    return f
