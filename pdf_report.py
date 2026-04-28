import io
import os
import datetime
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from cost_estimator import CostEstimate, format_mad, get_confidence_badge

# ── Arabic font registration ──────────────────────────────────────────────────
ARABIC_FONT      = "Helvetica"
ARABIC_FONT_BOLD = "Helvetica-Bold"

for font_path, font_name in [
    (r"C:\Windows\Fonts\arial.ttf",   "Arial"),
    (r"C:\Windows\Fonts\tahoma.ttf",  "Tahoma"),
    (r"C:\Windows\Fonts\times.ttf",   "Times"),
]:
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            ARABIC_FONT      = font_name
            ARABIC_FONT_BOLD = font_name  # same font, Arabic fonts rarely have separate bold
            break
        except Exception:
            continue

# ── Arabic text reshaper ──────────────────────────────────────────────────────
try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False

def prepare(text: str, lang: str) -> str:
    """Reshape and reorder Arabic text for correct PDF rendering."""
    if lang == 'ar' and ARABIC_SUPPORT:
        try:
            return get_display(reshape(str(text)))
        except Exception:
            return str(text)
    return str(text)

# ── Brand colors ──────────────────────────────────────────────────────────────
BRAND_DARK   = colors.HexColor("#1A2340")
BRAND_ACCENT = colors.HexColor("#E84C3D")
SEVERITY_COLORS = {
    "Léger":  colors.HexColor("#2ECC71"),
    "Moyen":  colors.HexColor("#F39C12"),
    "Sévère": colors.HexColor("#E74C3C"),
    "خفيف":   colors.HexColor("#2ECC71"),
    "متوسط":  colors.HexColor("#F39C12"),
    "شديد":   colors.HexColor("#E74C3C"),
}

# ── PDF string translations ───────────────────────────────────────────────────
PDF_STRINGS = {
    'fr': {
        'report_title':      "Rapport d'expertise automobile",
        'ref':               'Réf',
        'date':              'Date',
        'vehicle_info':      'Informations véhicule',
        'visual_analysis':   'Analyse visuelle',
        'original_image':    'Image originale',
        'gradcam_caption':   "Carte Grad-CAM (zones d'attention)",
        'severity_verdict':  'Verdict de sévérité',
        'severity_detected': 'Sévérité détectée',
        'model_confidence':  'Confiance du modèle',
        'ai_model':          'Modèle IA utilisé',
        'prob_by_class':     'Probabilités par classe :',
        'leger':             'Léger',
        'moyen':             'Moyen',
        'severe':            'Sévère',
        'cost_title':        'Estimation des coûts de réparation',
        'cost_low':          'Fourchette basse',
        'cost_avg':          'Estimation moyenne',
        'cost_high':         'Fourchette haute',
        'repair_time':       'Délai estimé',
        'recommendations':   'Recommandations :',
        'disclaimer': (
            "⚠️  Ce rapport est généré automatiquement par un système d'intelligence artificielle à titre indicatif. "
            "L'estimation de coût est une fourchette approximative basée sur des données moyennes du marché marocain. "
            "Elle ne constitue pas un devis officiel. Une expertise physique par un carrossier agréé reste indispensable "
            "avant toute prise en charge par l'assurance."
        ),
    },
    'ar': {
        'report_title':      'تقرير خبرة السيارة',
        'ref':               'المرجع',
        'date':              'التاريخ',
        'vehicle_info':      'معلومات المركبة',
        'visual_analysis':   'التحليل البصري',
        'original_image':    'الصورة الأصلية',
        'gradcam_caption':   'خريطة Grad-CAM (مناطق الاهتمام)',
        'severity_verdict':  'حكم الخطورة',
        'severity_detected': 'الخطورة المكتشفة',
        'model_confidence':  'ثقة النموذج',
        'ai_model':          'نموذج الذكاء الاصطناعي المستخدم',
        'prob_by_class':     'الاحتمالات لكل فئة :',
        'leger':             'خفيف',
        'moyen':             'متوسط',
        'severe':            'شديد',
        'cost_title':        'تقدير تكاليف الإصلاح',
        'cost_low':          'الحد الأدنى',
        'cost_avg':          'التقدير المتوسط',
        'cost_high':         'الحد الأقصى',
        'repair_time':       'المدة المقدرة',
        'recommendations':   'التوصيات :',
        'disclaimer': (
            "⚠️  هذا التقرير مُنشأ تلقائياً بواسطة نظام ذكاء اصطناعي على سبيل الإرشاد فقط. "
            "تقدير التكلفة هو نطاق تقريبي مبني على بيانات متوسط السوق المغربي. "
            "لا يُعدّ هذا عرض سعر رسمياً. تبقى الخبرة الفعلية من قِبل حداد معتمد ضرورية "
            "قبل أي تغطية من قِبل شركة التأمين."
        ),
    }
}

def p(key, lang):
    return PDF_STRINGS[lang].get(key, PDF_STRINGS['fr'].get(key, key))

# ── Helpers ───────────────────────────────────────────────────────────────────
def numpy_to_rl_image(arr: np.ndarray, max_width: float, max_height: float) -> RLImage:
    pil_img = Image.fromarray(
        (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    )
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    rl_img = RLImage(buf, width=max_width, height=max_height)
    rl_img.hAlign = "CENTER"
    return rl_img

# ── Main report generator ─────────────────────────────────────────────────────
def generate_report(
    original_img: np.ndarray,
    gradcam_img: np.ndarray,
    severity_label: str,
    probabilities: dict,
    confidence: float,
    cost_estimate: CostEstimate,
    vehicle_info: dict = None,
    claim_id: str = None,
    lang: str = 'fr',
) -> bytes:

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    styles = getSampleStyleSheet()

    # Font selection based on language
    font      = ARABIC_FONT      if lang == 'ar' else "Helvetica"
    font_bold = ARABIC_FONT_BOLD if lang == 'ar' else "Helvetica-Bold"
    align     = TA_RIGHT         if lang == 'ar' else TA_LEFT

    def style(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_style   = style("T",  fontSize=20, textColor=BRAND_DARK,              fontName=font_bold, spaceAfter=2)
    h2_style      = style("H2", fontSize=13, textColor=BRAND_DARK,              fontName=font_bold, spaceBefore=10, spaceAfter=4, alignment=align)
    body_style    = style("B",  fontSize=9,  textColor=colors.HexColor("#333333"), fontName=font,  leading=14, alignment=align)
    bold_body     = style("BB", fontSize=9,  textColor=BRAND_DARK,              fontName=font_bold, alignment=align)
    caption_style = style("C",  fontSize=8,  textColor=colors.HexColor("#777777"), fontName=font,  alignment=TA_CENTER)
    ref_style     = style("R",  fontSize=8,  textColor=colors.HexColor("#555555"), fontName=font,  alignment=TA_RIGHT)
    disclaimer_style = style("D", fontSize=7.5, textColor=colors.HexColor("#888888"), fontName=font, leading=11, alignment=align)

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    date_str  = datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")
    claim_str = claim_id or f"SIN-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    header_data = [[
        Paragraph(f"<b>{prepare(p('report_title', lang), lang)}</b>", title_style),
        Paragraph(
            f"{prepare(p('ref', lang), lang)} : {claim_str}<br/>"
            f"{prepare(p('date', lang), lang)} : {date_str}",
            ref_style
        )
    ]]
    header_table = Table(header_data, colWidths=[120*mm, 60*mm])
    header_table.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW",    (0,0), (-1,0),  1, BRAND_ACCENT),
        ("BOTTOMPADDING",(0,0), (-1,0),  6),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8*mm))

    # ── Vehicle info ──────────────────────────────────────────────────────────
    if vehicle_info:
        story.append(Paragraph(prepare(p('vehicle_info', lang), lang), h2_style))
        vdata = [
            [Paragraph(f"<b>{prepare(k, lang)}</b>", bold_body),
             Paragraph(prepare(str(v), lang), body_style)]
            for k, v in vehicle_info.items()
        ]
        vtable = Table(vdata, colWidths=[50*mm, 130*mm])
        vtable.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (0,-1), colors.HexColor("#F5F5F5")),
            ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ]))
        story.append(vtable)
        story.append(Spacer(1, 6*mm))

    # ── Images ────────────────────────────────────────────────────────────────
    story.append(Paragraph(prepare(p('visual_analysis', lang), lang), h2_style))
    img_w, img_h = 85*mm, 70*mm
    orig_rl    = numpy_to_rl_image(original_img, img_w, img_h)
    gradcam_rl = numpy_to_rl_image(gradcam_img,  img_w, img_h)

    img_data = [
        [orig_rl, gradcam_rl],
        [Paragraph(prepare(p('original_image', lang), lang), caption_style),
         Paragraph(prepare(p('gradcam_caption', lang), lang), caption_style)]
    ]
    img_table = Table(img_data, colWidths=[90*mm, 90*mm])
    img_table.setStyle(TableStyle([
        ("ALIGN",  (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (1,0), (-1,-1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 6*mm))

    # ── Severity verdict ──────────────────────────────────────────────────────
    story.append(Paragraph(prepare(p('severity_verdict', lang), lang), h2_style))
    sev_color     = SEVERITY_COLORS.get(severity_label, colors.gray)
    conf_label, _ = get_confidence_badge(confidence, lang)

    verdict_data = [
        [Paragraph(f"<b>{prepare(p('severity_detected', lang), lang)}</b>", bold_body),
         Paragraph(f"<b>{prepare(severity_label.upper(), lang)}</b>",
                   style("SV", fontSize=12, fontName=font_bold, textColor=sev_color))],
        [Paragraph(f"<b>{prepare(p('model_confidence', lang), lang)}</b>", bold_body),
         Paragraph(prepare(f"{confidence*100:.1f}% ({conf_label})", lang), body_style)],
        [Paragraph(f"<b>{prepare(p('ai_model', lang), lang)}</b>", bold_body),
         Paragraph("ResNet50V2 — best_model_final2.h5", body_style)],
    ]
    verdict_table = Table(verdict_data, colWidths=[60*mm, 120*mm])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (0,-1), colors.HexColor("#F5F5F5")),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 4*mm))

    # Probabilities
    story.append(Paragraph(f"<b>{prepare(p('prob_by_class', lang), lang)}</b>", bold_body))
    prob_keys = {
        "leger":  p('leger', lang),
        "moyen":  p('moyen', lang),
        "severe": p('severe', lang),
    }
    prob_data = [
        [Paragraph(prepare(prob_keys[k], lang), body_style),
         Paragraph(f"{v*100:.1f}%", bold_body)]
        for k, v in probabilities.items()
    ]
    prob_table = Table(prob_data, colWidths=[50*mm, 30*mm])
    prob_table.setStyle(TableStyle([
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#EEEEEE")),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 6*mm))

    # ── Cost estimate ─────────────────────────────────────────────────────────
    story.append(Paragraph(prepare(p('cost_title', lang), lang), h2_style))
    story.append(Paragraph(prepare(cost_estimate.description, lang), body_style))
    story.append(Spacer(1, 3*mm))

    cost_data = [
        [Paragraph(f"<b>{prepare(p('cost_low',  lang), lang)}</b>", bold_body),
         Paragraph(format_mad(cost_estimate.min_cost), body_style)],
        [Paragraph(f"<b>{prepare(p('cost_avg',  lang), lang)}</b>", bold_body),
         Paragraph(f"<b>{format_mad(cost_estimate.avg_cost)}</b>",
                   style("EM", fontSize=10, fontName=font_bold, textColor=BRAND_DARK))],
        [Paragraph(f"<b>{prepare(p('cost_high', lang), lang)}</b>", bold_body),
         Paragraph(format_mad(cost_estimate.max_cost), body_style)],
        [Paragraph(f"<b>{prepare(p('repair_time', lang), lang)}</b>", bold_body),
         Paragraph(prepare(cost_estimate.repair_time, lang), body_style)],
    ]
    cost_table = Table(cost_data, colWidths=[60*mm, 120*mm])
    cost_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (0,-1), colors.HexColor("#F5F5F5")),
        ("BACKGROUND",   (0,1), (1,1),  colors.HexColor("#FFF8E1")),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(cost_table)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph(f"<b>{prepare(p('recommendations', lang), lang)}</b>", bold_body))
    for rec in cost_estimate.recommendations:
        story.append(Paragraph(f"• {prepare(rec, lang)}", body_style))
    story.append(Spacer(1, 6*mm))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(prepare(p('disclaimer', lang), lang), disclaimer_style))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()