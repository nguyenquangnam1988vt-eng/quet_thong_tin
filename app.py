import streamlit as st
import pandas as pd
import easyocr
import re
from io import BytesIO
from PIL import Image
import numpy as np

# --------------------- CÀI ĐẶT BAN ĐẦU ---------------------
st.set_page_config(page_title="OCR Info Extractor", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['vi', 'en'], gpu=False)

reader = load_ocr()

# --------------------- KHỞI TẠO SESSION STATE ---------------------
if 'fields' not in st.session_state:
    st.session_state.fields = []          # [{'id': int, 'name': str, 'keyword': str, 'keep': bool}]
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'field_counter' not in st.session_state:
    st.session_state.field_counter = 0
if 'detected_fields' not in st.session_state:
    st.session_state.detected_fields = []

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
    """
    Nhận đầu vào có thể là:
    - UploadedFile (từ st.camera_input hoặc st.file_uploader)
    - PIL Image
    - numpy array
    - bytes
    Trả về văn bản đã nhận diện.
    """
    if image_input is None:
        return ""
    try:
        # Nếu là UploadedFile (có getvalue)
        if hasattr(image_input, 'getvalue'):
            bytes_data = image_input.getvalue()
            img = Image.open(BytesIO(bytes_data))
            img = np.array(img)
        # Nếu là PIL Image
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        # Nếu đã là numpy array
        elif isinstance(image_input, np.ndarray):
            img = image_input
        # Nếu là bytes
        elif isinstance(image_input, bytes):
            img = np.array(Image.open(BytesIO(image_input)))
        else:
            # Thử ép thành bytes
            img = np.array(Image.open(BytesIO(image_input)))
        # EasyOCR yêu cầu ảnh dạng numpy (RGB)
        if len(img.shape) == 3 and img.shape[2] == 4:
            # Nếu có kênh alpha, chuyển sang RGB
            img = img[:, :, :3]
        result = reader.readtext(img, detail=0)
        return "\n".join(result)
    except Exception as e:
        st.error(f"Lỗi khi nhận diện chữ: {str(e)}")
        return ""

def auto_detect_fields(text):
    """
    Tìm các dòng có dạng 'Tên trường: giá trị' hoặc 'Tên trường  giá trị'
    Trả về list các dict {name, keyword, sample_value}
    """
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
    # Loại bỏ trùng lặp
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

# --------------------- GIAO DIỆN CHÍNH ---------------------
st.title("📄 Trích xuất thông tin từ ảnh chụp")
st.markdown("---")

# Sidebar: Quản lý cấu hình trường
with st.sidebar:
    st.header("⚙️ Quản lý trường thông tin")
    
    # Hiển thị các trường vừa phát hiện (nếu có)
    if st.session_state.detected_fields:
        st.subheader("🔍 Các trường vừa phát hiện từ ảnh")
        st.info("Chọn các trường bạn muốn giữ lại, có thể sửa tên trường.")
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
        if st.button("💾 Lưu các trường đã chọn vào cấu hình"):
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
    
    # Thêm trường thủ công
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

    # Danh sách trường hiện có
    st.subheader("📋 Danh sách trường đã lưu")
    if not st.session_state.fields:
        st.info("Chưa có trường nào. Hãy chụp ảnh để tự động phát hiện hoặc thêm thủ công.")
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

# Main area
tab1, tab2, tab3 = st.tabs(["📸 Chụp ảnh / Upload", "📊 Bảng dữ liệu", "📥 Xuất Excel"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        option = st.radio("Chọn nguồn ảnh:", ("📷 Chụp từ camera", "📁 Tải ảnh lên"))
        image = None
        if option == "📷 Chụp từ camera":
            st.caption("💡 Trên điện thoại, bạn có thể chuyển sang camera sau bằng nút đổi camera (thường là biểu tượng mũi tên vòng) trên màn hình camera.")
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
        with st.spinner("Đang nhận diện chữ..."):
            text = ocr_image(image)
        st.text_area("📝 Văn bản nhận diện", text, height=150)

        # Nếu chưa có trường nào và chưa có detected_fields, tự động phát hiện
        if not st.session_state.fields and not st.session_state.detected_fields:
            detected = auto_detect_fields(text)
            if detected:
                st.session_state.detected_fields = detected
                st.info(f"Đã phát hiện {len(detected)} trường. Vui lòng vào sidebar để chọn trường cần giữ.")
                st.experimental_rerun()
            else:
                st.warning("Không phát hiện trường nào từ ảnh. Hãy thêm trường thủ công trong sidebar.")
        else:
            # Nếu có trường (đã lưu hoặc đã phát hiện) thì trích xuất
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
                    st.info("Không có trường nào được kích hoạt để trích xuất (kiểm tra các trường có được 'Giữ' không).")
            else:
                st.info("Đã phát hiện trường, hãy vào sidebar để chọn và lưu lại.")

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

# --------------------- HƯỚNG DẪN ---------------------
with st.expander("📖 Hướng dẫn sử dụng"):
    st.markdown("""
    **1. Chọn camera**:
    - Nếu dùng camera trên app, mặc định có thể là camera trước. Trên màn hình camera của trình duyệt, thường có nút chuyển đổi camera (hình mũi tên vòng) – bấm vào đó để dùng camera sau.
    - Nếu không thấy nút, bạn có thể dùng chức năng "Tải ảnh lên" – chụp ảnh bằng ứng dụng camera mặc định (chọn camera sau) và tải lên.

    **2. Tự động phát hiện trường**:
    - Khi chưa có trường nào, sau khi OCR, hệ thống sẽ phát hiện các dòng có dấu hai chấm và hiển thị trong sidebar để bạn chọn.

    **3. Các thao tác khác**:
    - Thêm/sửa/xóa trường thủ công trong sidebar.
    - Sau khi trích xuất, sửa giá trị (nếu sai) rồi thêm vào bảng.
    - Xuất Excel và chia sẻ qua Zalo (tải về và gửi file).
    """)
