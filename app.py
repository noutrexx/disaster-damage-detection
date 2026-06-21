import io
import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from core import (
    DAMAGE_COLORS,
    DAMAGE_LABELS_TR,
    average_confidence,
    count_by_class,
    create_demo_predictions,
    extract_predictions,
    predictions_to_csv,
    severity_index,
    severity_label,
)

load_dotenv()

APP_DIR = Path(__file__).parent
SAMPLE_DIR = APP_DIR / "sample_data"

SAMPLE_IMAGES = {
    "Ornek mahalle goruntusu": SAMPLE_DIR / "sample_satellite_like.png",
    "Yogun hasar testi": SAMPLE_DIR / "sample_heavy_damage.png",
    "Karisik hasar testi": SAMPLE_DIR / "sample_mixed_damage.png",
}


def html(markup: str) -> None:
    """Cok satirli HTML'i tek satira indirip Streamlit'e basar.

    Girintili satirlar Markdown tarafindan kod blogu sanilmasin diye her satir
    kirpilir ve birlestirilir.
    """
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


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


def _load_font(size: int) -> ImageFont.ImageFont:
    """Etiketler icin okunabilir bir font yuklemeye calisir."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_predictions(image: Image.Image, predictions: list[dict]) -> Image.Image:
    output = image.convert("RGB").copy()
    # Buyuk gorsellerde kutu/yazi orantili kalsin diye olcek hesaplanir.
    scale = max(1.0, min(output.width, output.height) / 420)
    line_width = max(2, int(round(3 * scale)))
    font = _load_font(max(12, int(round(13 * scale))))
    draw = ImageDraw.Draw(output)

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
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)

        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        pad = max(2, int(round(3 * scale)))
        tag_top = max(0, top - text_h - 2 * pad)
        draw.rectangle(
            (left, tag_top, left + text_w + 2 * pad, tag_top + text_h + 2 * pad),
            fill=color,
        )
        draw.text((left + pad, tag_top + pad), label, fill="black", font=font)

    return output


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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


def metric_cards(counts: dict[str, int]) -> None:
    """Sinif bazli sayaclar icin renkli kartlar cizer."""
    cards = []
    for class_name, color in DAMAGE_COLORS.items():
        cards.append(
            f"""
            <div class="metric-card" style="border-top: 4px solid {color};">
                <div class="metric-dot" style="background:{color};"></div>
                <div class="metric-value">{counts[class_name]}</div>
                <div class="metric-label">{DAMAGE_LABELS_TR[class_name]}</div>
                <div class="metric-sub">{class_name}</div>
            </div>
            """
        )
    html(f'<div class="metric-grid">{"".join(cards)}</div>')


def distribution_bars(counts: dict[str, int]) -> None:
    """Sinif dagilimini orantili yatay barlarla gosterir."""
    total = sum(counts.values()) or 1
    rows = []
    for class_name, color in DAMAGE_COLORS.items():
        value = counts[class_name]
        pct = 100 * value / total
        rows.append(
            f"""
            <div class="bar-row">
                <div class="bar-name">{DAMAGE_LABELS_TR[class_name]}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
                </div>
                <div class="bar-value">{value}</div>
            </div>
            """
        )
    html(f'<div class="bar-wrap">{"".join(rows)}</div>')


def severity_panel(counts: dict[str, int], avg_conf: float) -> None:
    score = severity_index(counts)
    label, color = severity_label(score)
    total = sum(counts.values())
    html(
        f"""
        <div class="severity-card">
            <div class="severity-head">
                <span>Hasar Siddet Endeksi</span>
                <span class="severity-badge" style="background:{color}1a; color:{color}; border:1px solid {color};">{label}</span>
            </div>
            <div class="severity-score" style="color:{color};">{score:.0f}<span>/100</span></div>
            <div class="severity-track">
                <div class="severity-fill" style="width:{score:.1f}%; background:{color};"></div>
            </div>
            <div class="severity-foot">
                <span>Toplam tespit: <b>{total}</b></span>
                <span>Ortalama guven: <b>{avg_conf:.0%}</b></span>
            </div>
        </div>
        """
    )


def color_legend() -> None:
    chips = "".join(
        f'<span class="legend-chip"><span class="legend-dot" style="background:{color};"></span>'
        f"{DAMAGE_LABELS_TR[name]}</span>"
        for name, color in DAMAGE_COLORS.items()
    )
    html(f'<div class="legend">{chips}</div>')


def render_prediction_panel(image: Image.Image, predictions: list[dict], source_label: str):
    counts = count_by_class(predictions)
    avg_conf = average_confidence(predictions)
    annotated = draw_predictions(image, predictions)

    severity_panel(counts, avg_conf)
    metric_cards(counts)

    image_col, chart_col = st.columns([3, 2], gap="large")
    with image_col:
        st.markdown('<div class="panel-title">Model ciktisi</div>', unsafe_allow_html=True)
        color_legend()
        st.image(annotated, use_container_width=True)
        st.caption(f"Sonuc kaynagi: {source_label}")
    with chart_col:
        st.markdown('<div class="panel-title">Sinif dagilimi</div>', unsafe_allow_html=True)
        distribution_bars(counts)
        st.markdown('<div class="panel-title">Disa aktar</div>', unsafe_allow_html=True)
        download_cols = st.columns(2)
        download_cols[0].download_button(
            "Gorseli indir",
            data=image_to_png_bytes(annotated),
            file_name="oahts_annotated.png",
            mime="image/png",
            use_container_width=True,
        )
        download_cols[1].download_button(
            "CSV indir",
            data=predictions_to_csv(predictions),
            file_name="oahts_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Tahmin JSON"):
        st.code(json.dumps(predictions, indent=2, ensure_ascii=False), language="json")


st.set_page_config(
    page_title="O-AHTS | Afet Hasar Tespit",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; max-width: 1200px; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #0f2440 0%, #15243d 55%, #1c2c3f 100%);
        border: 1px solid #24344f;
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(2, 8, 23, 0.45);
    }
    .hero-top { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .hero-logo { font-size: 34px; line-height: 1; }
    .hero h1 { margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.5px; color: #f8fafc; }
    .hero-sub { color: #93a4c0; margin-top: 6px; font-size: 1rem; }
    .hero-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
    .hero-badge {
        font-size: 0.78rem; padding: 4px 11px; border-radius: 999px;
        background: rgba(56, 189, 248, 0.12); color: #7dd3fc; border: 1px solid #1e4e6b;
    }
    .hero-badge.warn { background: rgba(247, 127, 0, 0.12); color: #fbbf24; border-color: #7c5a18; }

    /* Metric cards */
    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 6px 0 18px; }
    .metric-card {
        position: relative; background: #131c2e; border: 1px solid #24344f;
        border-radius: 12px; padding: 16px 16px 14px;
    }
    .metric-dot { width: 10px; height: 10px; border-radius: 50%; position: absolute; top: 16px; right: 16px; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #f1f5f9; line-height: 1.1; }
    .metric-label { color: #cbd5e1; font-size: 0.95rem; font-weight: 600; margin-top: 2px; }
    .metric-sub { color: #64748b; font-size: 0.74rem; font-family: monospace; margin-top: 2px; }

    /* Severity card */
    .severity-card {
        background: #131c2e; border: 1px solid #24344f; border-radius: 12px;
        padding: 16px 18px; margin-bottom: 4px;
    }
    .severity-head { display: flex; justify-content: space-between; align-items: center;
        color: #cbd5e1; font-weight: 600; font-size: 0.95rem; }
    .severity-badge { font-size: 0.75rem; padding: 3px 10px; border-radius: 999px; font-weight: 700; }
    .severity-score { font-size: 2.6rem; font-weight: 800; margin: 6px 0 8px; line-height: 1; }
    .severity-score span { font-size: 1rem; color: #64748b; font-weight: 600; }
    .severity-track { height: 9px; background: #1e293b; border-radius: 999px; overflow: hidden; }
    .severity-fill { height: 100%; border-radius: 999px; }
    .severity-foot { display: flex; justify-content: space-between; color: #94a3b8;
        font-size: 0.85rem; margin-top: 10px; }
    .severity-foot b { color: #e2e8f0; }

    /* Distribution bars */
    .panel-title { color: #e2e8f0; font-weight: 700; font-size: 1.02rem; margin: 4px 0 10px; }
    .bar-wrap { display: flex; flex-direction: column; gap: 11px; }
    .bar-row { display: grid; grid-template-columns: 96px 1fr 28px; align-items: center; gap: 10px; }
    .bar-name { color: #cbd5e1; font-size: 0.85rem; }
    .bar-track { height: 14px; background: #1e293b; border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; min-width: 2px; transition: width .3s ease; }
    .bar-value { color: #e2e8f0; font-weight: 700; text-align: right; font-size: 0.9rem; }

    /* Legend */
    .legend { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
    .legend-chip { display: inline-flex; align-items: center; gap: 6px; color: #94a3b8; font-size: 0.82rem; }
    .legend-dot { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }

    /* Status box */
    .status-box {
        border: 1px solid #7c5a18; border-left: 4px solid #f59e0b; border-radius: 10px;
        padding: 12px 16px; background: rgba(245, 158, 11, 0.08); margin-bottom: 16px;
        color: #fcd9a3; font-size: 0.9rem;
    }

    /* Roadmap list */
    .roadmap { list-style: none; padding: 0; margin: 0; }
    .roadmap li { padding: 7px 0; color: #cbd5e1; border-bottom: 1px solid #1e293b; font-size: 0.95rem; }
    .roadmap li .tick { color: #2a9d8f; font-weight: 700; margin-right: 8px; }
    .roadmap li .todo { color: #64748b; font-weight: 700; margin-right: 8px; }

    .app-footer { color: #64748b; font-size: 0.82rem; text-align: center;
        margin-top: 28px; padding-top: 14px; border-top: 1px solid #1e293b; }

    @media (max-width: 720px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
    </style>
    """,
    unsafe_allow_html=True,
)

mode_connected = bool(os.getenv("ROBOFLOW_API_KEY"))
connection_badge = (
    '<span class="hero-badge">Roboflow baglantisi hazir</span>'
    if mode_connected
    else '<span class="hero-badge warn">Demo modu (API key yok)</span>'
)

html(
    f"""
    <div class="hero">
        <div class="hero-top">
            <div class="hero-logo">🛰️</div>
            <div>
                <h1>O-AHTS · Afet Hasar Tespit Sistemi</h1>
                <div class="hero-sub">xView2/xBD uydu goruntuleri uzerinde hasar tespiti ve test arayuzu</div>
            </div>
        </div>
        <div class="hero-badges">
            <span class="hero-badge">xView2 / xBD</span>
            <span class="hero-badge">Roboflow Inference</span>
            <span class="hero-badge">4 hasar sinifi</span>
            {connection_badge}
        </div>
    </div>
    """
)

html(
    """
    <div class="status-box">
    ⚠️ Bu proje <b>gelistirme asamasindadir</b>. Resmi afet yonetimi, AFAD bildirimi veya saha karari
    icin kullanilmaz. Model egitimi bir kere yapilip, uygulamada hazir modelin test ve tahmin akisi gosterilir.
    </div>
    """
)

with st.sidebar:
    st.markdown("### ⚙️ Model Ayarlari")
    default_mode_index = 1 if os.getenv("ROBOFLOW_API_KEY") else 0
    mode = st.radio("Calisma modu", ["Demo", "Roboflow"], index=default_mode_index)

    if mode == "Demo":
        st.info("Demo modu: sabit ornek tahminlerle arayuz gosterilir.")
    else:
        st.success("Roboflow modu: hosted model veya workflow kullanilir.")

    roboflow_method = st.radio(
        "Roboflow tipi",
        ["Hosted model", "Workflow"],
        index=0,
        disabled=mode != "Roboflow",
    )
    st.divider()
    st.markdown("#### 🎚️ Filtreler")
    confidence_filter = st.slider("Minimum guven", 0.0, 1.0, 0.50, 0.05)
    selected_classes = st.multiselect(
        "Gosterilecek siniflar",
        list(DAMAGE_COLORS),
        default=list(DAMAGE_COLORS),
    )

    with st.expander("🔌 Roboflow baglantisi", expanded=mode == "Roboflow"):
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

    st.divider()
    st.caption("O-AHTS · ogrenci prototipi")


def apply_filters(predictions: list[dict]) -> list[dict]:
    return [
        item
        for item in predictions
        if float(item.get("confidence", 0)) >= confidence_filter
        and item.get("class") in selected_classes
    ]


tabs = st.tabs(["🔍 Tahmin", "🧪 Test", "📋 Proje Notlari"])

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
        predictions = apply_filters(predictions)
        render_prediction_panel(image, predictions, source_label)

with tabs[1]:
    st.subheader("Test Verileri ile Kontrol")
    st.write("Bu sekme hazir test gorselleri uzerinden modelin tahminlerini hizlica kontrol eder.")
    test_sample = st.selectbox("Test icin gorsel sec", list(SAMPLE_IMAGES), key="test_sample")
    test_image, test_path = load_sample_image(test_sample)

    if st.button("▶ Testi Calistir", type="primary"):
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
        predictions = apply_filters(predictions)
        render_prediction_panel(test_image, predictions, source_label)
    else:
        st.image(test_image, caption="Secilen test gorseli", use_container_width=True)

    st.code(
        "python scripts/test_model.py --image sample_data/sample_mixed_damage.png --model-id xview2-xbd/2",
        language="bash",
    )

with tabs[2]:
    st.subheader("Proje Kapsami")
    st.markdown(
        """
        Bu calisma SRS dokumanindaki buyuk O-AHTS fikrinin basitlestirilmis halidir.
        Amac, hazir/egitilmis bir modeli kullanarak test ve tahmin akisini gostermektir.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ✅ Su an hazir")
        html(
            """
            <ul class="roadmap">
                <li><span class="tick">✓</span> Streamlit tabanli modern arayuz</li>
                <li><span class="tick">✓</span> Roboflow hosted model / workflow baglantisi</li>
                <li><span class="tick">✓</span> Hazir test gorselleri ile tek gorsel tahmini</li>
                <li><span class="tick">✓</span> Sinif bazli sayaclar ve siddet endeksi</li>
                <li><span class="tick">✓</span> Sonuclari PNG / CSV / JSON disa aktarma</li>
            </ul>
            """
        )
    with col_b:
        st.markdown("#### 🚧 Sonraki asama")
        html(
            """
            <ul class="roadmap">
                <li><span class="todo">○</span> AFAD / Kandilli entegrasyonu</li>
                <li><span class="todo">○</span> Otomatik deprem tetikleme</li>
                <li><span class="todo">○</span> Uydu verisini otomatik indirme</li>
                <li><span class="todo">○</span> Harita / GeoJSON uretimi</li>
                <li><span class="todo">○</span> Saha ekibi mobil modulu ve bildirim</li>
            </ul>
            """
        )

    st.info(
        "Model egitimi uygulama icinde degil, ayri script ile bir kere yapilir. "
        "Detaylar icin README ve docs/O-AHTS-SRS.pdf dosyasina bakin."
    )

st.markdown(
    '<div class="app-footer">O-AHTS · Afet Hasar Tespit Sistemi · xView2/xBD · '
    "Roboflow Inference ile · ogrenci prototipi</div>",
    unsafe_allow_html=True,
)
