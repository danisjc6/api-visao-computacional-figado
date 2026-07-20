import os
import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Detecção de Fígado",
    layout="centered"
)

st.title("Detecção de Fígado Canino/Felino 🐶🐱")

default_api_url = os.environ.get(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)

api_url = st.sidebar.text_input(
    "URL da API",
    value=default_api_url
).rstrip("/")

uploaded_file = st.file_uploader(
    "Escolha uma imagem",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Imagem original",
        use_container_width=True
    )

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    try:

        response = requests.post(
            f"{api_url}/detectron/predict_auto",
            files=files
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.RequestException as e:

        st.error(f"Erro na comunicação com a API:\n{e}")

    else:

        if data.get("status") != "ok":

            st.warning(
                data.get(
                    "motivo",
                    "Erro desconhecido."
                )
            )

        else:

            st.success(
                f"✅ Espécie: {data['especie']} "
                f"(confiança: {data['confidence_especie']:.3f})"
            )

            st.write(
                f"**Número de estruturas detectadas:** "
                f"{data['num_instancias']}"
            )

            st.subheader("Detecções")

            for det in data["deteccoes"]:

                st.write(
                    f"**{det['classe']}** "
                    f"| Score: {det['score']:.3f}"
                )

            # -----------------------------
            # Reconstrói a imagem Base64
            # -----------------------------
            imagem_base64 = data.get("imagem_anotada")

            if imagem_base64:

                image_bytes = base64.b64decode(imagem_base64)

                annotated_img = Image.open(
                    BytesIO(image_bytes)
                )

                st.image(
                    annotated_img,
                    caption="Imagem anotada",
                    use_container_width=True
                )

            else:

                st.warning(
                    "A API não retornou a imagem anotada."
                )