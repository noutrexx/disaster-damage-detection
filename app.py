import json
import os
import random
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont


load_dotenv()

APP_DIR = Path(__file__).parent
SAMPLE_DIR = APP_DIR / "sample_data"

DAMAGE_COLORS = {
    "destroyed": "#d90429",
    "major-damage": "#f77f00",
    "minor-damage": "#fcbf49",
    "no-damage": "#2a9d8f",
}

SAMPLE_IMAGES = {
    "Ornek mahalle goruntusu": SAMPLE_DIR / "sample_satellite_like.png",
    "Yogun hasar testi": SAMPLE_DIR / "sample_heavy_damage.png",
    "Karisik hasar testi": SAMPLE_DIR / "sample_mixed_damage.png",
}


def create_demo_predictions(width: int, height: int) -> list[dict]:
    """API key olmadan arayuzu gostermek icin sahte ama sabit tahmin uretir."""
    random.seed(width * 1000 + height)
    classes = list(DAMAGE_COLORS)
    predictions = []

    for index in range(8):
        box_w = random.randint(max(35, width // 14), max(45, width // 7))
        box_h = random.randint(max(35, height // 14), max(45, height // 7))
        x = random.randint(box_w // 2, max(box_w // 2, width - box_w // 2))
        y = random.randint(box_h // 2, max(box_h // 2, height - box_h // 2))
        class_name = classes[index % len(classes)]

        predictions.append(
            {
                "class": class_name,
                "confidence": round(random.uniform(0.55, 0.93), 2),
                "x": x,
                "y": y,
                "width": box_w,
                "height": box_h,
            }
        )

    return predictions


def extract_predictions(result) -> list[dict]:
    if isinstance(result, dict) and "predictions" in result:
        return result["predictions"]

    if isinstance(result, list):
        for item in result:
            predictions = extract_predictions(item)
            if predictions:
                return predictions

    if isinstance(result, dict):
        for value in result.values():
            predictions = extract_predictions(value)
            if predictions:
                return predictions

    return []


def run_roboflow_model(image_path: str, api_key: str, model_id: str) -> list[dict]:
    api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
    model_id = model_id or os.getenv("ROBOFLOW_MODEL_ID", "")

    if not api_key or not model_id:
        raise RuntimeError("API key veya model id eksik.")

    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )
    result = client.infer(image_path, model_id=model_id)
    return extract_predictions(result)


def run_roboflow_workflow(
    image_path: str,
    api_key: str,
    workspace: str,
    workflow_id: str,
    classes: list[str],
) -> list[dict]:
    api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
    workspace = workspace or os.getenv("ROBOFLOW_WORKSPACE", "")
    workflow_id = workflow_id or os.getenv("ROBOFLOW_WORKFLOW_ID", "")

    if not api_key or not workspace or not workflow_id:
        raise RuntimeError("API key, workspace veya workflow id eksik.")

    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )
    result = client.run_workflow(
        workspace_name=workspace,
        workflow_id=workflow_id,
        images={"image": image_path},
        parameters={"classes": ",".join(classes)},
        use_cache=True,
    )
    return extract_predictions(result)


def draw_predictions(image: Image.Image, predictions: list[dict]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()

    for prediction in predictions:
        class_name = prediction.get("class", "unknown")
        color = DAMAGE_COLORS.get(class_name, "#ffffff")
        x = float(prediction.get("x", 0))
        y = float(prediction.get("y", 0))
        width = float(prediction.get("width", 0))
        height = float(prediction.get("height", 0))
        confidence = float(prediction.get("confidence", 0))

        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        label = f"{class_name} {confidence:.0%}"
        draw.rectangle((left, top, right, bottom), outline=color, width=4)
        draw.rectangle((left, max(0, top - 18), left + 130, top), fill=color)
        draw.text((left + 4, max(0, top - 15)), label, fill="black", font=font)

    return output


def count_by_class(predictions: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in DAMAGE_COLORS}
    for prediction in predictions:
        class_name = prediction.get("class")
        if class_name in counts:
            counts[class_name] += 1
    return counts


def save_uploaded_image(uploaded_file) -> tuple[Image.Image, str]:
    image = Image.open(uploaded_file).convert("RGB")
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    image.save(temp_file.name)
    return image, temp_file.name


def load_sample_image(sample_name: str) -> tuple[Image.Image, str]:
    image_path = SAMPLE_IMAGES[sample_name]
    image = Image.open(image_path).convert("RGB")
    return image, str(image_path)


def get_current_image(source: str, selected_sample: str, uploaded_file):
    if source == "Hazir test gorseli":
        return load_sample_image(selected_sample)
    if uploaded_file:
        return save_uploaded_image(uploaded_file)
    return None, None


def predict_image(
    image_path: str,
    image_size: tuple[int, int],
    mode: str,
    roboflow_method: str,
    api_key: str,
    model_id: str,
    workspace: str,
    workflow_id: str,
    selected_classes: list[str],
) -> tuple[list[dict], str]:
    if mode == "Demo":
        return create_demo_predictions(*image_size), "Demo modu"

    try:
        if roboflow_method == "Hosted model":
            predictions = run_roboflow_model(image_path, api_key, model_id)
            return predictions, f"Roboflow hosted model: {model_id}"

        predictions = run_roboflow_workflow(
            image_path=image_path,
            api_key=api_key,
            workspace=workspace,
            workflow_id=workflow_id,
            classes=selected_classes,
        )
        return predictions, f"Roboflow workflow: {workflow_id}"
    except Exception as exc:
        st.warning(f"Roboflow calismadi, demo sonuc gosteriliyor: {exc}")
        return create_demo_predictions(*image_size), "Demo fallback"


def render_prediction_panel(image: Image.Image, predictions: list[dict], source_label: str):
    counts = count_by_class(predictions)
    annotated = draw_predictions(image, predictions)

    metric_cols = st.columns(4)
    for col, class_name in zip(metric_cols, DAMAGE_COLORS):
        col.metric(class_name, counts[class_name])

    st.caption(f"Sonuc kaynagi: {source_label}")
    st.image(annotated, caption="Model ciktisi", use_container_width=True)

    with st.expander("Tahmin JSON"):
        st.code(json.dumps(predictions, indent=2, ensure_ascii=False), language="json")


st.set_page_config(page_title="O-AHTS", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; }
    .status-box {
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 12px 14px;
        background: #f6f8fa;
        margin-bottom: 12px;
    }
    .small-note { color: #57606a; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("O-AHTS Afet Hasar Tespit Sistemi")
st.caption("xView2/xBD verileri ile model egitimi, test ve basit tahmin arayuzu.")
st.markdown(
    """
    <div class="status-box">
    Bu proje su an <b>gelistirme asamasindadir</b>. Resmi afet yonetimi, AFAD bildirimi
    veya saha karari icin kullanilmaz. Amac ogrenci projesi olarak veri, model ve arayuz
    akisini gostermektir.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Ayarlari")
    default_mode_index = 1 if os.getenv("ROBOFLOW_API_KEY") else 0
    mode = st.radio("Calisma modu", ["Demo", "Roboflow"], index=default_mode_index)
    roboflow_method = st.radio(
        "Roboflow tipi",
        ["Hosted model", "Workflow"],
        index=0,
        disabled=mode != "Roboflow",
    )
    confidence_filter = st.slider("Minimum guven", 0.0, 1.0, 0.50, 0.05)
    selected_classes = st.multiselect(
        "Siniflar",
        list(DAMAGE_COLORS),
        default=list(DAMAGE_COLORS),
    )

    with st.expander("Roboflow baglantisi", expanded=mode == "Roboflow"):
        roboflow_api_key = st.text_input(
            "API key",
            value=os.getenv("ROBOFLOW_API_KEY", ""),
            type="password",
        )
        roboflow_model_id = st.text_input(
            "Model ID",
            value=os.getenv("ROBOFLOW_MODEL_ID", "xview2-xbd/2"),
        )
        roboflow_workspace = st.text_input(
            "Workspace",
            value=os.getenv("ROBOFLOW_WORKSPACE", "flow-wnra9"),
        )
        roboflow_workflow = st.text_input(
            "Workflow ID",
            value=os.getenv("ROBOFLOW_WORKFLOW_ID", ""),
        )

tabs = st.tabs(["Tahmin", "Model Egitimi", "Test", "Proje Notlari"])

with tabs[0]:
    st.subheader("Tek Gorsel Tahmini")
    image_source = st.radio(
        "Gorsel kaynagi",
        ["Hazir test gorseli", "Dosya yukle"],
        horizontal=True,
        key="predict_source",
    )
    selected_sample = st.selectbox("Test gorseli", list(SAMPLE_IMAGES), key="predict_sample")
    uploaded_file = None
    if image_source == "Dosya yukle":
        uploaded_file = st.file_uploader(
            "Uydu veya hava goruntusu yukle",
            type=["jpg", "jpeg", "png"],
            key="predict_upload",
        )

    image, image_path = get_current_image(image_source, selected_sample, uploaded_file)
    if image is None or image_path is None:
        st.info("Bir gorsel yukleyin veya hazir test gorseli secin.")
    else:
        predictions, source_label = predict_image(
            image_path=image_path,
            image_size=(image.width, image.height),
            mode=mode,
            roboflow_method=roboflow_method,
            api_key=roboflow_api_key,
            model_id=roboflow_model_id,
            workspace=roboflow_workspace,
            workflow_id=roboflow_workflow,
            selected_classes=selected_classes,
        )
        predictions = [
            item for item in predictions if float(item.get("confidence", 0)) >= confidence_filter
        ]
        render_prediction_panel(image, predictions, source_label)

with tabs[1]:
    st.subheader("Model Egitimi Plani")
    st.write(
        "Bu bolum model egitim surecini takip etmek icin hazirlandi. Gercek egitim "
        "Roboflow'dan indirilen YOLO formatli xView2/xBD verisi ile yapilir."
    )

    col_a, col_b, col_c = st.columns(3)
    epochs = col_a.number_input("Epoch", min_value=1, max_value=300, value=30, step=5)
    image_size = col_b.selectbox("Image size", [640, 768, 1024], index=0)
    base_model = col_c.selectbox("Baslangic modeli", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    dataset_yaml = st.text_input("Dataset YAML yolu", value="datasets/xview2-xbd/data.yaml")

    train_command = (
        f"python scripts/train_yolo.py --data {dataset_yaml} --model {base_model} "
        f"--epochs {epochs} --imgsz {image_size}"
    )
    st.code(train_command, language="bash")

    st.markdown(
        """
        Egitim adimlari:

        1. Roboflow Universe uzerinden xView2/xBD verisini YOLO formatinda indir.
        2. Indirilen klasoru `datasets/xview2-xbd/` altina koy.
        3. Yukaridaki komutu terminalde calistir.
        4. Egitilen agirlik dosyasini `runs/detect/train/weights/best.pt` olarak kullan.
        """
    )

with tabs[2]:
    st.subheader("Test Verileri ile Kontrol")
    st.write("Bu sekme hazir test gorselleri uzerinden modelin tahminlerini hizlica kontrol eder.")
    test_sample = st.selectbox("Test icin gorsel sec", list(SAMPLE_IMAGES), key="test_sample")
    test_image, test_path = load_sample_image(test_sample)

    if st.button("Testi Calistir", type="primary"):
        predictions, source_label = predict_image(
            image_path=test_path,
            image_size=(test_image.width, test_image.height),
            mode=mode,
            roboflow_method=roboflow_method,
            api_key=roboflow_api_key,
            model_id=roboflow_model_id,
            workspace=roboflow_workspace,
            workflow_id=roboflow_workflow,
            selected_classes=selected_classes,
        )
        predictions = [
            item for item in predictions if float(item.get("confidence", 0)) >= confidence_filter
        ]
        render_prediction_panel(test_image, predictions, source_label)
    else:
        st.image(test_image, caption="Secilen test gorseli", use_container_width=True)

    st.code(
        "python scripts/test_model.py --image sample_data/sample_mixed_damage.png --model-id xview2-xbd/2",
        language="bash",
    )

with tabs[3]:
    st.subheader("Proje Kapsami")
    st.markdown(
        """
        Bu calisma SRS dokumanindaki buyuk O-AHTS fikrinin basitlestirilmis halidir.

        - Su an sadece model egitimi, test ve tek gorsel tahmini hedefleniyor.
        - AFAD/Kandilli entegrasyonu yok.
        - Harita, GeoJSON, saha ekibi ve bildirim modulleri sonraki asamaya birakildi.
        - Roboflow hosted model ile hizli test yapilabiliyor.
        - Yerel egitim icin YOLO tabanli scriptler eklendi.
        """
    )
