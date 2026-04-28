TRANSLATIONS = {
    'fr': {
        # App title & intro
        'app_title': '🚗 InsurTech — Évaluation Automatique des Dommages',
        'app_subtitle': 'Uploadez **1 à 5 photos** du véhicule endommagé pour obtenir une analyse complète.',
        'upload_label': 'Choisir des images',
        'max_photos_warning': 'Maximum 5 photos. Seules les 5 premières seront analysées.',

        # Per-photo section
        'per_photo_title': '📸 Analyse par photo',
        'photo_label': 'Photo',

        # Consolidated result
        'consolidated_title': '📊 Résultat consolidé',
        'consolidated_label': 'Dommage consolidé',
        'based_on': 'Basé sur la moyenne de {} photo(s)',
        'avg_probs_title': 'Probabilités moyennes par classe',

        # Severity labels
        'leger': 'Léger',
        'moyen': 'Moyen',
        'severe': 'Sévère',

        # Fraud risk
        'risk_title': '🔍 Indicateur de risque',
        'risk_label': 'Risque',
        'risk_low': 'Faible',
        'risk_moderate': 'Modéré',
        'risk_high': 'Élevé',
        'risk_msg_low': 'Aucun signal suspect détecté.',
        'risk_msg_moderate': 'Vérification manuelle recommandée.',
        'risk_msg_high': 'Dossier à soumettre à un expert humain.',
        'inconsistent_warning': '⚠️ Photos incohérentes : {}',
        'suspicious_warning': '⚠️ Sévérité élevée avec faible confiance.',

        # Grad-CAM
        'gradcam_title': '🔥 Grad-CAM représentatif (photo la plus sévère)',
        'original_image': 'Image originale',
        'gradcam_map': 'Carte Grad-CAM',
        'explanation_label': '🧠 Explication',

        # Cost
        'cost_title': '💰 Estimation des coûts de réparation',
        'cost_low': 'Fourchette basse',
        'cost_avg': 'Estimation moyenne',
        'cost_high': 'Fourchette haute',
        'repair_time_label': '⏱️ Délai estimé : **{}** — Confiance IA : **{:.1f}% ({})**',
        'recommendations_label': '📋 Voir les recommandations',

        # Vehicle info
        'pdf_title': '📄 Générer un rapport PDF',
        'vehicle_info_label': '🚙 Ajouter les informations du véhicule (optionnel)',
        'brand': 'Marque',
        'year': 'Année',
        'model_car': 'Modèle',
        'plate': 'Immatriculation',
        'generate_pdf': '🖨️ Générer le rapport PDF',
        'generating': 'Génération du rapport en cours...',
        'pdf_success': '✅ Rapport généré avec succès !',
        'download_pdf': '⬇️ Télécharger le rapport PDF',
        'uncertainty_banner': '⚠️ Analyse incertaine — expertise humaine recommandée',
    },
    'ar': {
        'app_title': '🚗 InsurTech — التقييم التلقائي للأضرار',
        'app_subtitle': 'قم بتحميل **1 إلى 5 صور** للمركبة المتضررة للحصول على تحليل شامل.',
        'upload_label': 'اختر الصور',
        'max_photos_warning': 'الحد الأقصى 5 صور. سيتم تحليل أول 5 صور فقط.',

        'per_photo_title': '📸 التحليل لكل صورة',
        'photo_label': 'صورة',

        'consolidated_title': '📊 النتيجة الموحدة',
        'consolidated_label': 'الضرر الموحد',
        'based_on': 'بناءً على متوسط {} صورة',
        'avg_probs_title': 'متوسط الاحتمالات لكل فئة',

        'leger': 'خفيف',
        'moyen': 'متوسط',
        'severe': 'شديد',

        'risk_title': '🔍 مؤشر المخاطر',
        'risk_label': 'مخاطر',
        'risk_low': 'منخفضة',
        'risk_moderate': 'معتدلة',
        'risk_high': 'مرتفعة',
        'risk_msg_low': 'لم يتم اكتشاف أي إشارة مشبوهة.',
        'risk_msg_moderate': 'يُنصح بالتحقق اليدوي.',
        'risk_msg_high': 'يجب إحالة الملف إلى خبير بشري.',
        'inconsistent_warning': '⚠️ صور غير متسقة : {}',
        'suspicious_warning': '⚠️ خطورة عالية مع ثقة منخفضة.',

        'gradcam_title': '🔥 خريطة Grad-CAM التمثيلية (أشد صورة)',
        'original_image': 'الصورة الأصلية',
        'gradcam_map': 'خريطة Grad-CAM',
        'explanation_label': '🧠 تفسير',

        'cost_title': '💰 تقدير تكاليف الإصلاح',
        'cost_low': 'الحد الأدنى',
        'cost_avg': 'التقدير المتوسط',
        'cost_high': 'الحد الأقصى',
        'repair_time_label': '⏱️ المدة المقدرة : **{}** — ثقة الذكاء الاصطناعي : **{:.1f}% ({})**',
        'recommendations_label': '📋 عرض التوصيات',

        'pdf_title': '📄 إنشاء تقرير PDF',
        'vehicle_info_label': '🚙 إضافة معلومات المركبة (اختياري)',
        'brand': 'العلامة التجارية',
        'year': 'السنة',
        'model_car': 'الموديل',
        'plate': 'رقم اللوحة',
        'generate_pdf': '🖨️ إنشاء تقرير PDF',
        'generating': 'جارٍ إنشاء التقرير...',
        'pdf_success': '✅ تم إنشاء التقرير بنجاح!',
        'download_pdf': '⬇️ تحميل تقرير PDF',
        'uncertainty_banner': '⚠️ تحليل غير مؤكد — يُنصح بمراجعة خبير بشري',
    }
}

def t(key, lang, *args):
    """Get translated string, optionally formatting with args."""
    text = TRANSLATIONS[lang].get(key, TRANSLATIONS['fr'].get(key, key))
    if args:
        return text.format(*args)
    return text