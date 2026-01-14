import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Detecção de Fígado", layout="centered")

st.title("Detecção de Fígado Canino/Felino 🐶🐱")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagem original
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem original", use_column_width=True)

    # Enviar para a API
    files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
    try:
        response = requests.post("http://127.0.0.1:8000/detectron/predict_auto", files=files)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na comunicação com a API: {e}")
    else:
        if data.get("status") != "ok":
            st.warning(data.get("motivo", "Erro desconhecido"))
        else:
            st.success(f"✅ Espécie detectada: {data['especie']} (confiança: {data['confidence_especie']})")
            st.write(f"Número de instâncias detectadas: {data['num_instancias']}")

            st.subheader("Detecções:")
            for det in data["deteccoes"]:
                st.write(f"- {det['classe']} | Score: {det['score']} | BBox: {det['bbox']}")

            # Mostrar imagem anotada
            annotated_path = data["imagem_anotada"]
            annotated_img = Image.open(annotated_path)
            st.image(annotated_img, caption="Imagem anotada", use_column_width=True)
