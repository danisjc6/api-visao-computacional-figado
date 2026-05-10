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
# IDs dos arquivos no Google Drive
# =========================
CLASSIFIER_ID="1B_Ohq7HqCkzCBIh8C5NsrFA6RAfFfwPt"
CANINO_ID="1f3rOxLYnwad-knkd8nZTslqdfQ2hChFM"
FELINO_ID="12r40vOmQZnbXkYMPqkYfE-fNJWonqUzG"

download_model() {
    local file_id="$1"
    local output_path="$2"

    gdown --id "$file_id" -O "$output_path"

    if [ ! -s "$output_path" ]; then
        echo "❌ Modelo vazio ou não baixado corretamente: $output_path"
        exit 1
    fi
}

# =========================
# Download
# =========================
echo "▶ Baixando classificador de espécie..."
download_model "$CLASSIFIER_ID" models/classifier/species_classifier.pth

echo "▶ Baixando modelo Detectron2 – Canino..."
download_model "$CANINO_ID" models/detectron/canino/model_final_canino.pth

echo "▶ Baixando modelo Detectron2 – Felino..."
download_model "$FELINO_ID" models/detectron/felino/model_final_felino.pth

echo "✅ Todos os modelos foram baixados com sucesso!"
