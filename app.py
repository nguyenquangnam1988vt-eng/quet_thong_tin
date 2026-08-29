import streamlit as st
import pandas as pd
import easyocr
import re
from io import BytesIO
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# --------------------- CÀI ĐẶT BAN ĐẦU ---------------------
st.set_page_config(page_title="OCR Info Extractor", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['vi', 'en'], gpu=False)

reader = load_ocr()

# --------------------- KHỞI TẠO SESSION STATE ---------------------
if 'fields' not in st.session_state:
    st.session_state.fields = []
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'field_counter' not in st.session_state:
    st.session_state.field_counter = 0
if 'detected_fields' not in st.session_state:
    st.session_state.detected_fields = []
if 'captured_frame' not in st.session_state:
    st.session_state.captured_frame = None  # lưu ảnh chụp từ webrtc

# --------------------- HÀM TIỆN ÍCH ---------------------
def add_field(name, keyword, keep=True):
    st.session_state.field_counter += 1
    st.session_state.fields.append({
        'id': st.session_state.field_counter,
        'name': name,
        'keyword': keyword,
        'keep': keep
    })

def remove_field(field_id):
    st.session_state.fields = [f for f in st.session_state.fields if f['id'] != field_id]

def update_field(field_id, name=None, keyword=None, keep=None):
    for f in st.session_state.fields:
        if f['id'] == field_id:
            if name is not None:
                f['name'] = name
            if keyword is not None:
                f['keyword'] = keyword
            if keep is not None:
                f['keep'] = keep
            break

def ocr_image(image_input):
    if image_input is None:
        return ""
    try:
        if hasattr(image_input, 'getvalue'):
            bytes_data = image_input.getvalue()
            img = Image.open(BytesIO(bytes_data))
            img = np.array(img)
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        elif isinstance(image_input, bytes):
            img = np.array(Image.open(BytesIO(image_input)))
        else:
            img = np.array(Image.open(BytesIO(image_input)))
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        result = reader.readtext(img, detail=0)
        return "\n".join(result)
    except Exception as e:
        st.error(f"Lỗi OCR: {str(e)}")
        return ""

def auto_detect_fields(text):
    lines = text.split('\n')
    detected = []
    pattern = re.compile(r'^(.+?)\s*[:：]\s*(.+)$', re.UNICODE)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            field_name = match.group(1).strip()
            value = match.group(2).strip()
            if len(field_name) < 50 and not re.search(r'[^\w\sÀ-ỹ]', field_name):
                detected.append({
                    'name': field_name,
                    'keyword': field_name + ':',
                    'sample_value': value[:50]
                })
    seen = set()
    unique = []
    for d in detected:
        if d['name'] not in seen:
            seen.add(d['name'])
            unique.append(d)
    return unique

def extract_values(text, fields):
    row = {}
    for f in fields:
        if f['keep']:
            pattern = re.compile(rf"{re.escape(f['keyword'])}\s*(.+)", re.IGNORECASE | re.UNICODE)
            match = pattern.search(text)
            if not match:
                pattern2 = re.compile(rf"{re.escape(f['keyword'])}\s+(.+)", re.IGNORECASE | re.UNICODE)
                match = pattern2.search(text)
            if match:
                row[f['name']] = match.group(1).strip()
            else:
                row[f['name']] = ""
    return row

# --------------------- CLASS XỬ LÝ VIDEO (chụp ảnh) ---------------------
class VideoCaptureProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        self.frame = img
        return frame

# --------------------- GIAO DIỆN CHÍNH ---------------------
st.title("📄 Trích xuất thông tin từ ảnh chụp")
st.markdown("---")

# Sidebar quản lý trường (giữ nguyên như code trước)
with st.sidebar:
    st.header("⚙️ Quản lý trường thông tin")
    # ... (phần này không đổi, copy từ code trước) ...
    # Để ngắn gọn, tôi sẽ bỏ qua phần lặp lại, nhưng bạn cần giữ nguyên phần sidebar từ code trước.

# Tạm thời tôi sẽ viết lại phần sidebar ngắn gọn ở đây, nhưng thực tế bạn nên giữ nguyên như cũ.
# (Vì dài, tôi sẽ chỉ giữ phần chính)

# Main area
tab1, tab2, tab3 = st.tabs(["📸 Chụp ảnh / Upload", "📊 Bảng dữ liệu", "📥 Xuất Excel"])

with tab1:
    st.subheader("Chọn nguồn ảnh")
    option = st.radio("Phương thức:", ("📷 Chụp từ camera (WebRTC)", "📁 Tải ảnh lên"))
    
    if option == "📷 Chụp từ camera (WebRTC)":
        # Sử dụng webrtc để chọn camera sau
        st.info("Bấm 'Start' để mở camera, chọn thiết bị camera sau trong danh sách (nếu có).")
        webrtc_ctx = webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoCaptureProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "facingMode": {"ideal": "environment"}  # ưu tiên camera sau
                },
                "audio": False
            },
            async_processing=True,
        )
        if webrtc_ctx.video_processor:
            frame = webrtc_ctx.video_processor.frame
            if frame is not None:
                st.image(frame, channels="BGR", caption="Luồng camera", use_container_width=True)
                if st.button("📸 Chụp ảnh từ luồng này"):
                    st.session_state.captured_frame = frame.copy()
                    st.success("Đã chụp ảnh!")
            else:
                st.warning("Chưa có hình ảnh từ camera.")
        else:
            st.warning("Camera chưa khởi tạo. Nhấn 'Start' để bắt đầu.")
        
        # Nếu đã có ảnh chụp từ trước, hiển thị và xử lý
        if st.session_state.captured_frame is not None:
            st.image(st.session_state.captured_frame, caption="Ảnh đã chụp", channels="BGR", use_container_width=True)
            # OCR và xử lý tương tự như với ảnh upload
            if st.button("🔍 Nhận diện chữ từ ảnh đã chụp"):
                with st.spinner("Đang OCR..."):
                    text = ocr_image(st.session_state.captured_frame)
                st.text_area("Văn bản nhận diện", text, height=150)
                # Xử lý phát hiện trường hoặc trích xuất (như code cũ)
                # (Phần này giống hệt như khi có ảnh từ upload)
                if not st.session_state.fields and not st.session_state.detected_fields:
                    detected = auto_detect_fields(text)
                    if detected:
                        st.session_state.detected_fields = detected
                        st.info(f"Đã phát hiện {len(detected)} trường. Vào sidebar để chọn.")
                    else:
                        st.warning("Không phát hiện trường nào.")
                else:
                    if st.session_state.fields:
                        row = extract_values(text, st.session_state.fields)
                        if row:
                            st.subheader("Thông tin trích xuất")
                            edited_row = {}
                            cols = st.columns(min(len(row), 4))
                            for i, (key, val) in enumerate(row.items()):
                                with cols[i % len(cols)]:
                                    edited_row[key] = st.text_input(key, value=val, key=f"edit_{key}_{i}")
                            if st.button("➕ Thêm vào bảng"):
                                if edited_row:
                                    new_df = pd.DataFrame([edited_row])
                                    st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
                                    st.success("Đã thêm!")
                                    st.balloons()
                                    st.session_state.captured_frame = None  # reset sau khi thêm
                                    st.experimental_rerun()
                    else:
                        st.info("Đã phát hiện trường, hãy vào sidebar để lưu lại.")
    
    else:  # Tải ảnh lên
        uploaded = st.file_uploader("Chọn ảnh (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Ảnh đã tải", use_container_width=True)
            with st.spinner("Đang OCR..."):
                text = ocr_image(uploaded)
            st.text_area("Văn bản nhận diện", text, height=150)
            # Xử lý tương tự như trên (có thể tách thành hàm riêng)

# Tab2 và tab3 giữ nguyên như cũ

# --------------------- HƯỚNG DẪN ---------------------
with st.expander("📖 Hướng dẫn sử dụng"):
    st.markdown("""
    **Chọn camera**: Khi dùng WebRTC, bạn có thể chọn camera sau bằng cách:
    - Nhấn "Start", trình duyệt sẽ hỏi quyền truy cập camera.
    - Trong danh sách thiết bị (hiển thị ở góc phải), chọn camera sau (thường có tên 'environment' hoặc 'back').
    - Nếu không chọn được, hãy dùng phương thức "Tải ảnh lên" để chụp ảnh bằng ứng dụng camera mặc định và tải lên.

    **Các bước khác** tương tự như hướng dẫn trước.
    """)
