# ?? InsurTech AI - Détection de dommages véhicule

## ?? Description du projet

Ce projet a pour objectif de développer un système automatique de classification de la sévérité des dommages automobiles (léger / moyen / sévère) à partir d'images, adapté au marché marocain de l'assurance.

L'application permet :
- Upload de 1 à 5 photos d'un véhicule endommagé
- Classification automatique de la sévérité
- Visualisation Grad-CAM (carte de chaleur explicative)
- Détection de fraude (incohérence entre photos)
- Estimation des coûts de réparation en MAD
- Génération de rapport PDF
- Interface bilingue Français / Arabe (avec support RTL)

## ??? Architecture technique

| Composant | Technologie |
|-----------|-------------|
| Deep Learning | TensorFlow / Keras |
| Modèle | ResNet50V2 (pré-entraîné ImageNet + fine-tuning) |
| Interface utilisateur | Streamlit |
| Traitement d'images | OpenCV, PIL |
| Génération PDF | ReportLab |
| Visualisation | Plotly, Matplotlib |
| Support arabe | Arabic-reshaper, Python-bidi |

## ?? Performances du modèle

| Métrique | Score |
|----------|-------|
| Validation accuracy | **77,13%** |
| Macro F1-Score | **80,2%** |

### Performances par classe

| Classe | Précision | Rappel | F1-Score |
|--------|-----------|--------|----------|
| Léger | 94,0% | 91,4% | **92,7%** |
| Moyen | 64,0% | 73,8% | **68,6%** |
| Sévère | 81,6% | 76,9% | **79,2%** |

## ?? Installation et exécution

### Prérequis
- Python 3.9 ou supérieur
- Pip (gestionnaire de paquets)

### Étapes d'installation

1. Cloner le dépôt :
```bash
git clone https://github.com/REHAB2911/car-damage-detection.git
cd car-damage-detection
## 📥 Téléchargement du modèle

Le modèle entraîné étant trop volumineux pour GitHub, téléchargez-le ici : 
[Télécharger best_model_final2.h5](https://drive.google.com/file/d/1bu-pvsxp4p2GJeJmMUgnfUSuLxL7f52F/view?usp=sharing)

Placez le fichier dans le dossier du projet avant de lancer l'application.


