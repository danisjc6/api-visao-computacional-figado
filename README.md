# 🧠 API de Visão Computacional – Classificação e Detecção de Fígado (Canino/Felino)

Este repositório contém uma **API em FastAPI** e **scripts auxiliares** para:

* Classificação de espécie (**canino x felino**) a partir de imagens de fígado
* Detecção de estruturas hepáticas usando **Detectron2**
* Avaliação científica dos modelos treinados

O projeto foi desenvolvido com foco em **pesquisa aplicada**, **reprodutibilidade** e **uso em produção**.

---

## 📁 Estrutura do Projeto

```text
api_visao_computacional/
│
├── app/                    # Código principal da API (FastAPI)
│   ├── main.py             # Endpoints da API
│   ├── classifier.py       # Classificador canino/felino (PyTorch)
│   ├── detectron.py        # Inferência Detectron2 (produção)
│   ├── utils.py            # Pré-processamento e utilidades
│   └── routers/            # Rotas adicionais
│
├── models/
│   ├── classifier/         # Pesos do classificador
│   └── detectron/          # Pesos e configs Detectron2 (canino/felino)
│
├── configs/                # Arquivos YAML de configuração
│
├── scripts/                # Scripts offline (avaliação e inferência)
│   ├── infer_detectron.py
│   ├── evaluate_classifier.py
│   ├── evaluate_detectron_coco.py
│   ├── evaluate_detectron_labelme.py
│   ├── predict_detectron_labelme.py
│   └── results/
│
├── venv/                   # Ambiente virtual
└── README.md               # Este arquivo
```

---

## 🚀 API (Produção)

### ▶️ Arquivo principal

**`app/main.py`**

* Inicializa o FastAPI
* Carrega os modelos uma única vez
* Expõe o endpoint principal:

```http
POST /predict
```

Fluxo do endpoint:

1. Recebe imagem
2. Classifica a espécie (canino/felino)
3. Executa Detectron2 com o modelo correspondente
4. Valida se há fígado
5. Retorna JSON + imagem anotada

---

## 🤖 Modelos

### Classificador (PyTorch)

* Arquivo: `app/classifier.py`
* Entrada: imagem
* Saída: espécie + confiança
* Modelo: ResNet treinada

Pesos:

```text
models/classifier/species_classifier.pth
```

---

### Detectron2 (Detecção)

* Arquivo: `app/detectron.py`
* Função principal: `load_predictor(especie)`
* Um modelo por espécie (canino/felino)

Pesos:

```text
models/detectron/canino/model_final_canino.pth
models/detectron/felino/model_final_felino.pth
```

⚠️ Este código é **somente inferência**, adequado para produção.

---

## 🧪 Scripts (`scripts/`)

### 🟢 `infer_detectron.py`

**Inferência offline** com Detectron2.

* Recebe imagens individuais ou pasta
* Salva imagens anotadas
* Não usa dataset

📌 Usado para testes manuais e depuração.

---

### 🟢 `evaluate_classifier.py`

Avaliação do **classificador canino/felino**.

* Usa dataset de validação
* Métricas:

  * Accuracy
  * Confusion Matrix

📌 Uso científico / relatório

---

### 🟢 `evaluate_detectron_coco.py`

Avaliação **oficial Detectron2 (COCO)**.

* Usa dataset no formato COCO
* Métricas:

  * mAP
  * AP50
  * AP75

📌 Usado para validação científica do modelo
📌 **Não usado em produção**

---

### 🟡 `evaluate_detectron_labelme.py`

Avaliação para datasets anotados com **LabelMe**.

* Converte LabelMe → Detectron2

📌 Use apenas se o dataset for LabelMe

---

### 🟡 `predict_detectron_labelme.py`

Inferência em **datasets LabelMe**.

* Gera imagens anotadas
* Uso offline

---

## ❌ O que NÃO vai para produção

* Scripts de avaliação
* Registro de datasets
* COCOEvaluator

> **Regra de ouro:** Avaliação ≠ Inferência

---

## ▶️ Como rodar a API

```bash
source venv/bin/activate
uvicorn app.main:app --reload
```

Acesse:

* Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 📊 Resultados

Resultados de avaliação e inferência são salvos em:

```text
scripts/results/
```

---

## 🧑‍🔬 Observação Final

Este projeto foi estruturado para:

* Pesquisa acadêmica
* Reprodutibilidade
* Uso em ambiente real (API)

Qualquer dúvida sobre avaliação, inferência ou deploy deve considerar essa separação.

---

📌 **Autora:** Daniela Oliveira
📌 **Área:** Visão Computacional aplicada à Anatomia Veterinária
