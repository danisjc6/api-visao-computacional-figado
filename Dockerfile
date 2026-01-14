FROM python:3.10-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN pip install --upgrade pip

# 1️⃣ Instala torch e torchvision (CPU)
RUN pip install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# 2️⃣ Copia requirements e instala dependências da API
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3️⃣ Clona Detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git

# 4️⃣ Instala Detectron2 SEM build isolation
RUN pip install -e detectron2 --no-build-isolation

# Copia o restante do projeto
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

