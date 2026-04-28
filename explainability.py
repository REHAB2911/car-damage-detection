import numpy as np

# Heatmap is 224x224 after resize. Divide into zones.
# Horizontal: left / center / right (thirds)
# Vertical:   top / middle / bottom (thirds)

def _get_region(heatmap: np.ndarray) -> tuple[str, str]:
    """Returns (horizontal_zone, vertical_zone) of peak activation."""
    h, w = heatmap.shape
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    if x < w / 3:
        horiz = 'left'
    elif x < 2 * w / 3:
        horiz = 'center'
    else:
        horiz = 'right'

    if y < h / 3:
        vert = 'top'
    elif y < 2 * h / 3:
        vert = 'middle'
    else:
        vert = 'bottom'

    return horiz, vert


def _coverage(heatmap: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of image with activation above threshold."""
    return float(np.mean(heatmap > threshold))


REGION_LABELS = {
    'fr': {
        ('left',   'top'):    'dans le coin avant gauche',
        ('center', 'top'):    'sur la partie avant centrale',
        ('right',  'top'):    'dans le coin avant droit',
        ('left',   'middle'): 'sur le flanc gauche',
        ('center', 'middle'): 'au centre du véhicule',
        ('right',  'middle'): 'sur le flanc droit',
        ('left',   'bottom'): 'dans le coin arrière gauche',
        ('center', 'bottom'): 'sur la partie arrière centrale',
        ('right',  'bottom'): 'dans le coin arrière droit',
    },
    'ar': {
        ('left',   'top'):    'في الزاوية الأمامية اليسرى',
        ('center', 'top'):    'في الجزء الأمامي المركزي',
        ('right',  'top'):    'في الزاوية الأمامية اليمنى',
        ('left',   'middle'): 'على الجانب الأيسر',
        ('center', 'middle'): 'في وسط المركبة',
        ('right',  'middle'): 'على الجانب الأيمن',
        ('left',   'bottom'): 'في الزاوية الخلفية اليسرى',
        ('center', 'bottom'): 'في الجزء الخلفي المركزي',
        ('right',  'bottom'): 'في الزاوية الخلفية اليمنى',
    }
}

SPREAD_LABELS = {
    'fr': {
        'focused':  'Les dommages semblent localisés.',
        'moderate': 'Les dommages sont modérément étendus.',
        'wide':     'Les dommages sont étendus sur une large zone.',
    },
    'ar': {
        'focused':  'يبدو أن الأضرار موضعية.',
        'moderate': 'الأضرار منتشرة بشكل معتدل.',
        'wide':     'الأضرار منتشرة على مساحة واسعة.',
    }
}

def get_explanation(heatmap: np.ndarray, lang: str = 'fr') -> str:
    """
    Returns a one-sentence plain-language explanation of where
    the model detected damage, based on Grad-CAM heatmap.
    """
    horiz, vert = _get_region(heatmap)
    coverage = _coverage(heatmap)
    region_label = REGION_LABELS[lang].get((horiz, vert), '')

    if coverage < 0.15:
        spread_key = 'focused'
    elif coverage < 0.35:
        spread_key = 'moderate'
    else:
        spread_key = 'wide'

    spread_label = SPREAD_LABELS[lang][spread_key]

    if lang == 'fr':
        return f"Le modèle a détecté des dommages concentrés {region_label} du véhicule. {spread_label}"
    else:
        return f"اكتشف النموذج أضراراً مركّزة {region_label} للمركبة. {spread_label}"