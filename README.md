# API Visão Computacional – Fígado Canino/Felino 🐶🐱

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-lightgrey)
![Detectron2](https://img.shields.io/badge/Detectron2-0.6-orange)

API para detecção e classificação de fígado canino e felino em imagens médicas, utilizando **FastAPI**, **PyTorch** e **Detectron2**. Inclui interface web via **Streamlit** e scripts de avaliação.

---

## 🗂 Estrutura do Projeto

```text
api_visao_computacional/
│
├─ app/                        # Código principal da API
│  ├─ __init__.py
│  ├─ main.py                  # Instancia FastAPI e endpoints
│  ├─ classifier.py            # Classificador de espécie
│  ├─ detectron.py             # Funções Detectron2
│  ├─ utils.py                 # Funções utilitárias (leitura de imagens, anotação etc.)
│  ├─ services/
│  │  ├─ __init__.py
│  │  └─ prediction_service.py # Pipeline único de predição
│  └─ routers/
│     ├─ __init__.py
│     └─ detectron.py          # Endpoints específicos Detectron
│
├─ configs/
│  └─ app.yaml                 # Configurações gerais da API (caminhos de modelos etc.)
│
├─ models/
│  ├─ classifier/
│  │  └─ species_classifier.pth
│  └─ detectron/
│     ├─ canino/
│     │  ├─ inferencia_canino.yaml
│     │  └─ model_final_canino.pth
│     └─ felino/
│        ├─ inferencia_felino.yaml
│        └─ model_final_felino.pth
│
├─ scripts/                    # Scripts auxiliares
│  ├─ evaluate_classifier.py
│  ├─ evaluate_detectron.py
│  ├─ evaluate_detectron_labelme.py
│  ├─ infer_detectron.py
│  ├─ predict_detectron_labelme.py
│  └─ download_models.sh       # Script para baixar modelos grandes
│
├─ uploads/                     # Imagens enviadas pelo usuário
├─ outputs/                     # Resultados anotados gerados
├─ app_streamlit.py             # Interface Streamlit
├─ Dockerfile
├─ .dockerignore
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## ⚡ Rodando Localmente

1. Crie e ative o ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate
```

2. Atualize o `pip` e instale as dependências:

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/detectron2.git"
```

> O Detectron2 pode variar conforme sistema operacional, versão do PyTorch e uso de CPU/GPU. Se a instalação acima falhar, consulte a documentação oficial do Detectron2 para escolher o comando adequado ao seu ambiente.

3. Baixe os modelos treinados:

```bash
bash scripts/download_models.sh
```

Os pesos `.pth` não são versionados no Git. Eles devem ficar nos caminhos abaixo:

```text
models/classifier/species_classifier.pth
models/detectron/canino/model_final_canino.pth
models/detectron/felino/model_final_felino.pth
```

4. Suba a API FastAPI:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Teste os endpoints:

- Health check: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs

5. Em outro terminal, com o ambiente virtual ativo, suba a interface Streamlit:

```bash
streamlit run app_streamlit.py
```

Por padrão, o Streamlit usa a API em `http://127.0.0.1:8000`. Para apontar para outra URL:

```bash
API_BASE_URL=http://127.0.0.1:8000 streamlit run app_streamlit.py
```

## 🐳 Usando Docker

Build da imagem:

```bash
docker build -t api-figado .
```

Rodar container:

```bash
docker run -p 8000:8000 api-figado
```

A API estará acessível em http://localhost:8000.

## 🧰 Scripts Auxiliares

| Script                         | Função                                           |
| ------------------------------ | ------------------------------------------------ |
| `download_models.sh`           | Baixa ou move pesos grandes para a pasta correta |
| `evaluate_classifier.py`       | Avalia o classificador CNN                       |
| `evaluate_detectron.py`        | Avalia modelos Detectron2 em formato COCO        |
| `evaluate_detectron_labelme.py` | Avalia modelos Detectron2 com anotações LabelMe  |
| `infer_detectron.py`           | Executa inferência em imagens de teste           |
| `predict_detectron_labelme.py` | Prediz imagens usando dataset LabelMe            |


## 📂 Estrutura de Modelos

Classificador CNN:

```text
models/classifier/species_classifier.pth
```

Detectron2:

```text
models/detectron/canino/model_final_canino.pth
models/detectron/felino/model_final_felino.pth
```

Configs YAML correspondentes em cada pasta.

## 🚀 Deploy

Pode ser feito em servidor Linux, VPS ou cloud (AWS, GCP, Azure) usando Docker.

Basta buildar a imagem no servidor e rodar o container.

Streamlit pode ser exposto em uma porta separada ou integrado à API com reverse proxy (Nginx).

## 🔧 Observações

Logs e resultados: salvos em outputs/ e logs/ (quando criado).

Uploads temporários: uploads/.

Git: arquivos pesados (.pth) podem ser tratados com download_models.sh ou Git LFS.

Reprodutibilidade: Docker garante ambiente consistente para qualquer servidor.

## 💡 Contato

Desenvolvido por Daniela Oliveira
daniela.oliveira@ufape.edu.br
