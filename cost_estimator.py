from dataclasses import dataclass
from typing import Tuple

@dataclass
class CostEstimate:
    severity: str
    min_cost: int
    max_cost: int
    avg_cost: int
    repair_time: str
    description: str
    recommendations: list[str]

COST_CONFIG = {
    "leger": {
        "label":        {"fr": "Léger",   "ar": "خفيف"},
        "min_cost":     500,
        "max_cost":     4000,
        "repair_time":  {"fr": "1 à 3 jours",  "ar": "1 إلى 3 أيام"},
        "description":  {
            "fr": "Dommages superficiels : éraflures, petites bosses, rayures de peinture.",
            "ar": "أضرار سطحية: خدوش وحفر صغيرة وخدوش في الطلاء."
        },
        "recommendations": {
            "fr": [
                "Réparation par débosselage sans peinture (PDR) si possible",
                "Retouche peinture localisée",
                "Pas de remplacement de pièce nécessaire en général",
                "Évaluation recommandée chez un carrossier agréé",
            ],
            "ar": [
                "الإصلاح بتقنية الدهن بدون طلاء (PDR) إن أمكن",
                "لمسة طلاء موضعية",
                "لا حاجة لاستبدال قطع بشكل عام",
                "يُنصح بتقييم لدى حداد معتمد",
            ]
        }
    },
    "moyen": {
        "label":        {"fr": "Moyen",   "ar": "متوسط"},
        "min_cost":     4000,
        "max_cost":     15000,
        "repair_time":  {"fr": "3 à 7 jours",  "ar": "3 إلى 7 أيام"},
        "description":  {
            "fr": "Dommages modérés : déformation de carrosserie, bris de vitres ou feux, impacts structurels légers.",
            "ar": "أضرار متوسطة: تشوه في الهيكل، كسر في الزجاج أو الأضواء، تأثيرات هيكلية خفيفة."
        },
        "recommendations": {
            "fr": [
                "Remplacement probable de 1 à 3 pièces de carrosserie",
                "Peinture complète du panneau concerné",
                "Vérifier les systèmes de sécurité (airbags, capteurs ADAS)",
                "Devis détaillé fortement recommandé",
            ],
            "ar": [
                "الاستبدال المحتمل لـ 1 إلى 3 قطع من الهيكل",
                "طلاء كامل للجزء المعني",
                "التحقق من أنظمة الأمان (الوسائد الهوائية، حساسات ADAS)",
                "يُنصح بشدة بالحصول على عرض سعر مفصل",
            ]
        }
    },
    "severe": {
        "label":        {"fr": "Sévère",  "ar": "شديد"},
        "min_cost":     15000,
        "max_cost":     80000,
        "repair_time":  {"fr": "7 à 30+ jours", "ar": "7 إلى 30+ يوماً"},
        "description":  {
            "fr": "Dommages graves : structure endommagée, multiples pièces à remplacer, véhicule potentiellement irréparable.",
            "ar": "أضرار جسيمة: هيكل تالف، قطع متعددة تحتاج إلى استبدال، المركبة قد تكون غير قابلة للإصلاح."
        },
        "recommendations": {
            "fr": [
                "Expertise approfondie indispensable avant toute réparation",
                "Risque de perte totale du véhicule (valeur réparation > valeur vénale)",
                "Vérification obligatoire du châssis et des trains roulants",
                "Déclaration sinistre à l'assurance recommandée immédiatement",
            ],
            "ar": [
                "خبرة معمقة ضرورية قبل أي إصلاح",
                "خطر الخسارة الكلية للمركبة (تكلفة الإصلاح > القيمة التجارية)",
                "الفحص الإلزامي للهيكل والمحاور",
                "يُنصح بالإبلاغ الفوري لشركة التأمين",
            ]
        }
    }
}

def get_cost_estimate(severity_class: str, confidence: float, lang: str = 'fr') -> CostEstimate:
    config = COST_CONFIG.get(severity_class, COST_CONFIG["moyen"])

    min_c = config["min_cost"]
    max_c = config["max_cost"]

    if confidence >= 0.80:
        margin = 0.10
    elif confidence >= 0.60:
        margin = 0.20
    else:
        margin = 0.30

    adjusted_min = int(min_c * (1 - margin * 0.3))
    adjusted_max = int(max_c * (1 + margin * 0.2))
    avg = (adjusted_min + adjusted_max) // 2

    return CostEstimate(
        severity=config["label"][lang],
        min_cost=adjusted_min,
        max_cost=adjusted_max,
        avg_cost=avg,
        repair_time=config["repair_time"][lang],
        description=config["description"][lang],
        recommendations=config["recommendations"][lang]
    )

def format_mad(amount: int) -> str:
    return f"{amount:,} MAD".replace(",", " ")

def get_confidence_badge(confidence: float, lang: str = 'fr') -> tuple[str, str]:
    labels = {
        'fr': [("Faible", "red"), ("Modérée", "orange"), ("Élevée", "green")],
        'ar': [("منخفضة", "red"), ("معتدلة", "orange"), ("مرتفعة", "green")],
    }
    if confidence >= 0.80:
        return labels[lang][2]
    elif confidence >= 0.60:
        return labels[lang][1]
    else:
        return labels[lang][0]