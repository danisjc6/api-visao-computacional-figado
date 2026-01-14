#!/bin/bash

set -e  # interrompe se algo der errado

echo "📥 Baixando modelos da API Visão Computacional – Fígado"
echo "------------------------------------------------------"

# =========================
# Criar diretórios
# =========================
mkdir -p models/classifier
mkdir -p models/detectron/canino
mkdir -p models/detectron/felino

# =========================
# URLs dos modelos
# =========================
CLASSIFIER_URL="https://drive.google.com/file/d/1B_Ohq7HqCkzCBIh8C5NsrFA6RAfFfwPt/view?usp=drive_link"
CANINO_URL="https://drive.google.com/file/d/1f3rOxLYnwad-knkd8nZTslqdfQ2hChFM/view?usp=drive_link"
FELINO_URL="https://drive.google.com/file/d/12r40vOmQZnbXkYMPqkYfE-fNJWonqUzG/view?usp=drive_link"

# =========================
# Download
# =========================
echo "▶ Baixando classificador de espécie..."
wget -O models/classifier/species_classifier.pth "$CLASSIFIER_URL"

echo "▶ Baixando modelo Detectron2 – Canino..."
wget -O models/detectron/canino/model_final_canino.pth "$CANINO_URL"

echo "▶ Baixando modelo Detectron2 – Felino..."
wget -O models/detectron/felino/model_final_felino.pth "$FELINO_URL"

echo "✅ Todos os modelos foram baixados com sucesso!"
