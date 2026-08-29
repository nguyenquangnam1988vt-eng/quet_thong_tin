import streamlit as st
import pandas as pd
import easyocr
import re
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

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

# --------------------- HÀM TIỀN XỬ LÝ ẢNH ---------------------
def preprocess_image(image_input):
    """
    Chuyển đổi đầu vào thành ảnh PIL, tiền xử lý: grayscale, tăng độ tương phản, resize.
    """
    if image_input is None:
        return None
    try:
        if hasattr(image_input, 'getvalue'):
            img = Image.open(BytesIO(image_input.getvalue()))
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            img = Image.open(BytesIO(image_input))
        # Chuyển sang grayscale
        if img.mode != 'L':
            img = img.convert('L')
        # Tăng độ tương phản
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
        # Resize để dễ OCR (giữ nguyên tỉ lệ, tối đa 1500px chiều dài)
        max_size = 1500
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        # Chuyển sang numpy
        img_np = np.array(img)
        return img_np
    except Exception as e:
        st.error(f"Lỗi tiền xử lý ảnh: {e}")
        return None

# --------------------- HÀM OCR CẢI TIẾN ---------------------
def ocr_image(image_input):
    img_np = preprocess_image(image_input)
    if img_np is None:
        return ""
    try:
        # Thử với detail=0 để lấy văn bản
        result = reader.readtext(img_np, detail=0, paragraph=False)
        full_text = "\n".join(result)
        return full_text
    except Exception as e:
        st.error(f"Lỗi OCR: {e}")
        return ""

# --------------------- PHÁT HIỆN TRƯỜNG CẢI TIẾN ---------------------
def auto_detect_fields(text):
    """
    Tìm tất cả các cụm có dạng 'tên trường: giá trị' hoặc 'tên trường : giá trị'
    Không yêu cầu ở đầu dòng.
    """
    if not text.strip():
        return []
    detected = []
    # Mẫu tìm kiếm: tên trường (không chứa dấu hai chấm) theo sau bởi dấu hai chấm và phần giá trị
    # Cho phép khoảng trắng xung quanh dấu hai chấm
    pattern = re.compile(r'([^:：\n]+?)\s*[:：]\s*([^\n]+)', re.UNICODE)
    matches = pattern.findall(text)
    for field_name, value in matches:
        field_name = field_name.strip()
        value = value.strip()
        if len(field_name) < 50 and not re.search(r'[^\w\sÀ-ỹ]', field_name):
            # Bỏ qua những dòng quá ngắn hoặc chỉ là số
            if len(value) > 0:
                detected.append({
                    'name': field_name,
                    'keyword': field_name + ':',
                    'sample_value': value[:50]
                })
    # Loại bỏ trùng tên (giữ lần xuất hiện đầu tiên)
    seen = set()
    unique = []
    for d in detected:
        if d['name'] not in seen:
            seen.add(d['name'])
            unique.append(d)
    return unique

# --------------------- CÁC HÀM QUẢN LÝ TRƯỜNG (KHÔNG ĐỔI) ---------------------
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

def extract_values(text, fields):
    row = {}
    for f in fields:
        if f['keep']:
            pattern = re.compile(rf"{re.escape(f['keyword'])}\s*(.+)", re.IGNORECASE | re.UNICODE)
            match = pattern.search(text)
            if not match:
                # Thử không dấu hai chấm nhưng có khoảng trắng
                pattern2 = re.compile(rf"{re.escape(f['keyword'])}\s+(.+)", re.IGNORECASE | re.UNICODE)
                match = pattern2.search(text)
            if match:
                row[f['name']] = match.group(1).strip()
            else:
                row[f['name']] = ""
    return row

# --------------------- GIAO DIỆN CHÍNH ---------------------
st.title("📄 Trích xuất thông tin từ ảnh chụp (cải tiến OCR)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Quản lý trường thông tin")
    if st.session_state.detected_fields:
        st.subheader("🔍 Các trường vừa phát hiện từ ảnh")
        st.info("Chọn các trường muốn giữ, sửa tên nếu cần.")
        detected_choices = []
        for idx, d in enumerate(st.session_state.detected_fields):
            col1, col2 = st.columns([1, 3])
            with col1:
                keep = st.checkbox("Giữ", value=True, key=f"detect_keep_{idx}")
            with col2:
                new_name = st.text_input("Tên trường", value=d['name'], key=f"detect_name_{idx}")
            detected_choices.append({
                'original': d,
                'keep': keep,
                'new_name': new_name,
                'keyword': new_name + ':'
            })
        if st.button("💾 Lưu các trường đã chọn"):
            for choice in detected_choices:
                if choice['keep']:
                    add_field(choice['new_name'], choice['keyword'], keep=True)
            st.session_state.detected_fields = []
            st.success(f"Đã thêm {len([c for c in detected_choices if c['keep']])} trường")
            st.experimental_rerun()
        if st.button("🗑️ Bỏ qua các trường phát hiện này"):
            st.session_state.detected_fields = []
            st.experimental_rerun()
        st.markdown("---")
    
    with st.expander("➕ Thêm trường thủ công", expanded=False):
        new_name = st.text_input("Tên trường")
        new_keyword = st.text_input("Từ khóa tìm kiếm (ví dụ: 'Họ tên:')")
        if st.button("Thêm trường"):
            if new_name and new_keyword:
                add_field(new_name, new_keyword)
                st.success(f"Đã thêm trường '{new_name}'")
                st.experimental_rerun()
            else:
                st.error("Vui lòng nhập đầy đủ")
    
    st.subheader("📋 Danh sách trường đã lưu")
    if not st.session_state.fields:
        st.info("Chưa có trường nào.")
    else:
        for f in st.session_state.fields:
            with st.container():
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    new_name = st.text_input(f"Tên #{f['id']}", value=f['name'], key=f"name_{f['id']}")
                with col2:
                    new_keyword = st.text_input(f"Từ khóa #{f['id']}", value=f['keyword'], key=f"kw_{f['id']}")
                with col3:
                    keep = st.checkbox("Giữ", value=f['keep'], key=f"keep_{f['id']}")
                col4, col5 = st.columns([1, 1])
                with col4:
                    if st.button("Cập nhật", key=f"update_{f['id']}"):
                        update_field(f['id'], name=new_name, keyword=new_keyword, keep=keep)
                        st.success("Đã cập nhật")
                        st.experimental_rerun()
                with col5:
                    if st.button("🗑️ Xóa", key=f"del_{f['id']}"):
                        remove_field(f['id'])
                        st.experimental_rerun()
                st.markdown("---")
        if st.button("🗑️ Xóa tất cả trường"):
            st.session_state.fields = []
            st.session_state.field_counter = 0
            st.experimental_rerun()
    
    st.markdown("---")
    if st.button("🔄 Reset toàn bộ dữ liệu"):
        st.session_state.clear()
        st.experimental_rerun()

# Main
tab1, tab2, tab3 = st.tabs(["📸 Chụp ảnh / Upload", "📊 Bảng dữ liệu", "📥 Xuất Excel"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        option = st.radio("Chọn nguồn ảnh:", ("📷 Chụp từ camera", "📁 Tải ảnh lên"))
        image = None
        if option == "📷 Chụp từ camera":
            st.caption("💡 Trên điện thoại, có thể chuyển sang camera sau bằng nút đổi camera trên màn hình.")
            image = st.camera_input("Chụp ảnh")
        else:
            uploaded = st.file_uploader("Chọn ảnh (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded)
    with col2:
        if image is not None:
            st.image(image, caption="Ảnh đã chọn", use_container_width=True)
        else:
            st.info("Chưa có ảnh")

    if image is not None:
        # Nút OCR
        if st.button("🔍 Nhận diện chữ (OCR)"):
            with st.spinner("Đang OCR..."):
                text = ocr_image(image)
            st.session_state.last_text = text  # lưu để debug
            st.text_area("📝 Văn bản nhận diện", text, height=200)

            # Debug: hiển thị số dòng
            lines = text.split('\n')
            st.caption(f"Số dòng OCR: {len(lines)}")
            # Hiển thị các dòng có dấu hai chấm để kiểm tra
            colon_lines = [l for l in lines if ':' in l or '：' in l]
            if colon_lines:
                st.caption("Các dòng có dấu hai chấm:")
                for l in colon_lines[:10]:
                    st.text(l)
            else:
                st.warning("Không tìm thấy dòng nào có dấu hai chấm. Kiểm tra lại ảnh hoặc thêm trường thủ công.")

            # Phát hiện trường
            if not st.session_state.fields and not st.session_state.detected_fields:
                detected = auto_detect_fields(text)
                if detected:
                    st.session_state.detected_fields = detected
                    st.info(f"Đã phát hiện {len(detected)} trường. Vào sidebar để chọn.")
                else:
                    st.warning("Không phát hiện trường nào. Hãy thêm trường thủ công trong sidebar.")
            else:
                if st.session_state.fields:
                    row = extract_values(text, st.session_state.fields)
                    if row:
                        st.subheader("🔎 Thông tin trích xuất")
                        edited_row = {}
                        cols = st.columns(min(len(row), 4))
                        for i, (key, val) in enumerate(row.items()):
                            with cols[i % len(cols)]:
                                edited_row[key] = st.text_input(f"{key}", value=val, key=f"edit_{key}_{i}")
                        if st.button("➕ Thêm vào bảng", use_container_width=True):
                            if edited_row:
                                new_df = pd.DataFrame([edited_row])
                                st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
                                st.success("✅ Đã thêm dữ liệu!")
                                st.balloons()
                                st.experimental_rerun()
                    else:
                        st.info("Không trích xuất được giá trị nào. Kiểm tra từ khóa trường.")
                else:
                    st.info("Đã có trường phát hiện nhưng chưa lưu. Vào sidebar lưu lại.")

with tab2:
    st.subheader("📊 Bảng dữ liệu đã thu thập")
    if not st.session_state.data.empty:
        st.dataframe(st.session_state.data, use_container_width=True)
        st.caption(f"Tổng số dòng: {len(st.session_state.data)}")
    else:
        st.info("Chưa có dữ liệu.")

with tab3:
    st.subheader("📥 Xuất dữ liệu ra Excel")
    if st.button("Tạo file Excel"):
        if st.session_state.data.empty:
            st.warning("Bảng dữ liệu trống!")
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.data.to_excel(writer, index=False, sheet_name='Data')
            st.download_button(
                label="⬇️ Tải file Excel",
                data=output.getvalue(),
                file_name="thong_tin_trich_xuat.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Hướng dẫn
with st.expander("📖 Hướng dẫn sử dụng"):
    st.markdown("""
    **Cải tiến OCR**:
    - Ảnh được tự động tăng độ tương phản và chuyển sang grayscale để dễ nhận diện.
    - Nếu OCR vẫn kém, hãy thử chụp ảnh rõ hơn, ánh sáng tốt, chữ to và thẳng hàng.
    - Sử dụng nút "Nhận diện chữ" sau khi chọn ảnh.

    **Phát hiện trường**:
    - Hệ thống tìm tất cả các dòng có dấu hai chấm (:) và hiển thị trong sidebar.
    - Nếu không phát hiện, bạn có thể thêm trường thủ công trong sidebar.

    **Các bước khác** như trước.
    """)
