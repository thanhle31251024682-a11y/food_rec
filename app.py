# ============================================================
#  Canteen Food Recognition – UEH  |  Streamlit app
#  app.py – chạy: streamlit run app.py
# ============================================================
import ast, re, json, base64, traceback
import numpy as np
from PIL import Image
import cv2
import streamlit as st

# ─── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Canteen UEH",
    page_icon="🍱",
    layout="wide",
)

# ─── Menu giá ────────────────────────────────────────────────────────────────
MENU = {
    "1. Cơm trắng":         {"price": 10000, "display": "Cơm trắng",         "emoji": "🍚"},
    "2. Đậu hũ sốt cà":     {"price": 25000, "display": "Đậu hũ sốt cà",     "emoji": "🟫"},
    "3. Cá hú kho":          {"price": 30000, "display": "Cá hú kho",          "emoji": "🐟"},
    "4. Thịt kho trứng":     {"price": 30000, "display": "Thịt kho trứng",     "emoji": "🥚"},
    "5. Thịt kho":           {"price": 25000, "display": "Thịt kho",           "emoji": "🥩"},
    "6. Canh chua có cá":    {"price": 25000, "display": "Canh chua (có cá)",  "emoji": "🍲"},
    "7. Canh chua không cá": {"price": 10000, "display": "Canh chua (ko cá)",  "emoji": "🥣"},
    "8. Sườn nướng":         {"price": 30000, "display": "Sườn nướng",         "emoji": "🍖"},
    "9. Canh rau":           {"price":  7000, "display": "Canh rau",           "emoji": "🥬"},
    "10. Rau xào":           {"price": 10000, "display": "Rau xào",            "emoji": "🥦"},
    "11. Trứng chiên":       {"price": 25000, "display": "Trứng chiên",        "emoji": "🍳"},
}

IMG_SIZE = (224, 224)

# ─── Hardcode 5 ngăn khay UEH (tỷ lệ 0.0–1.0) ───────────────────────────────
#  ┌─────────────────┬───────────┐
#  │    Ngăn 1       │  Ngăn 2   │
#  │    (trái trên)  ├───────────┤
#  │                 │  Ngăn 3   │
#  ├─────────────────┼───────────┤
#  │    Ngăn 5       │  Ngăn 4   │
#  │    (trái dưới)  │           │
#  └─────────────────┴───────────┘
TRAY_CELLS = [
    ("ngan_1", 0.04, 0.03, 0.60, 0.52),
    ("ngan_2", 0.62, 0.03, 0.98, 0.27),
    ("ngan_3", 0.62, 0.29, 0.98, 0.50),
    ("ngan_4", 0.62, 0.52, 0.98, 0.98),
    ("ngan_5", 0.04, 0.54, 0.60, 0.98),
]


# ─── Load model (cache – chỉ load 1 lần) ─────────────────────────────────────
def _parse_shape(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val.strip())
            return list(parsed) if isinstance(parsed, tuple) else parsed
        except Exception:
            pass
    return val


def _fix_cfg(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "DTypePolicy" and "config" in obj:
            return obj["config"].get("name", "float32")
        if obj.get("class_name") == "InputLayer":
            cfg = obj.get("config", {})
            if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                cfg["batch_input_shape"] = cfg.pop("batch_shape")
            if "batch_input_shape" in cfg:
                cfg["batch_input_shape"] = _parse_shape(cfg["batch_input_shape"])
            for bad in ["optional", "sparse", "ragged",
                        "quantization_config", "dtype_policy"]:
                cfg.pop(bad, None)
            if isinstance(cfg.get("dtype"), dict):
                cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")
        cfg2 = obj.get("config", {})
        if isinstance(cfg2, dict) and isinstance(cfg2.get("dtype"), dict):
            cfg2["dtype"] = cfg2["dtype"].get("config", {}).get("name", "float32")
        return {k: _fix_cfg(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_cfg(i) for i in obj]
    return obj


@st.cache_resource(show_spinner="Đang load model AI...")
def load_resources():
    import h5py
    import tensorflow as tf

    model, idx_to_class = None, {}
    try:
        with h5py.File("canteen_model.h5", "r") as f:
            raw = f.attrs.get("model_config")
            cfg_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        config = _fix_cfg(json.loads(cfg_str))
        model = tf.keras.models.model_from_json(json.dumps(config))
        model.load_weights("canteen_model.h5", by_name=False, skip_mismatch=False)
    except Exception as e:
        st.warning(f"⚠️ Không load được model: {e}")

    try:
        with open("class_indices.json", encoding="utf-8") as f:
            class_indices = json.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
    except Exception as e:
        st.warning(f"⚠️ Không đọc được class_indices.json: {e}")

    return model, idx_to_class


# ─── Crop 5 ngăn ─────────────────────────────────────────────────────────────
def get_crops(img_bgr: np.ndarray, padding: int = 12):
    H, W = img_bgr.shape[:2]
    crops = []
    for (label, rx1, ry1, rx2, ry2) in TRAY_CELLS:
        x1 = max(0, int(rx1 * W) + padding)
        y1 = max(0, int(ry1 * H) + padding)
        x2 = min(W, int(rx2 * W) - padding)
        y2 = min(H, int(ry2 * H) - padding)
        crops.append((label, img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)))
    return crops


# ─── Classify 1 crop ─────────────────────────────────────────────────────────
def classify(crop_bgr: np.ndarray, model, idx_to_class):
    if model is None:
        return "unknown", 0.0
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    arr = np.array(pil.resize(IMG_SIZE), dtype=np.float32) / 255.0
    pred = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    idx = int(np.argmax(pred))
    return idx_to_class.get(idx, "unknown"), float(pred[idx])


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🍱 AI Canteen UEH")
st.caption("Chụp ảnh khay cơm → nhận diện món → tính tiền tự động")

model, idx_to_class = load_resources()

uploaded = st.file_uploader(
    "📷 Tải ảnh khay cơm lên",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    # Đọc ảnh
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize nếu quá lớn
    H, W = img.shape[:2]
    if max(H, W) > 1024:
        s = 1024 / max(H, W)
        img = cv2.resize(img, (int(W * s), int(H * s)))

    with st.spinner("Đang phân tích..."):
        crops = get_crops(img)
        results = []
        debug = img.copy()
        COLORS = [
            (255, 80,  80),  (80, 180, 255), (80, 220, 80),
            (255, 165,  0),  (180,  0, 255), (0,  200, 200),
        ]

        for i, (label, crop, bbox) in enumerate(crops):
            cls, cf = classify(crop, model, idx_to_class)
            info = MENU.get(cls, {})
            results.append({
                "label":   label,
                "cls":     cls,
                "display": info.get("display", cls),
                "emoji":   info.get("emoji", "🍽️"),
                "price":   info.get("price", 0),
                "conf":    cf,
                "crop":    crop,
                "bbox":    bbox,
            })
            # Vẽ debug
            x1, y1, x2, y2 = bbox
            col = COLORS[i % len(COLORS)]
            cv2.rectangle(debug, (x1, y1), (x2, y2), col, 3)
            txt = f"N{i+1} {info.get('display', cls)} {cf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(debug, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
            cv2.putText(debug, txt, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # ── Layout: ảnh gốc | ảnh debug ──────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Ảnh gốc")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_right:
        st.subheader("Kết quả nhận diện")
        st.image(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.divider()

    # ── Kết quả từng ngăn ────────────────────────────────────────────────────
    st.subheader("📋 Chi tiết từng ngăn")
    cols = st.columns(5)
    for i, r in enumerate(results):
        with cols[i]:
            crop_rgb = cv2.cvtColor(r["crop"], cv2.COLOR_BGR2RGB)
            st.image(crop_rgb, use_container_width=True)
            st.markdown(f"**Ngăn {i+1}**")
            st.markdown(f"{r['emoji']} {r['display']}")
            st.progress(r["conf"])
            st.caption(f"Độ tin cậy: {r['conf']*100:.1f}%")
            if r["price"] > 0:
                st.markdown(f"💰 **{r['price']:,}đ**")
            else:
                st.caption("Không tính tiền")

    st.divider()

    # ── Hoá đơn tổng ─────────────────────────────────────────────────────────
    st.subheader("🧾 Hoá đơn")
    total = 0
    for i, r in enumerate(results):
        if r["price"] > 0:
            c1, c2, c3 = st.columns([1, 4, 2])
            c1.write(f"Ngăn {i+1}")
            c2.write(f"{r['emoji']} {r['display']}")
            c3.write(f"**{r['price']:,}đ**")
            total += r["price"]

    st.divider()
    st.markdown(f"### Tổng cộng: **{total:,}đ**")

    if not model:
        st.error("Model chưa load được – kết quả đều là unknown. "
                 "Kiểm tra lại file canteen_model.h5 và requirements.txt.")
