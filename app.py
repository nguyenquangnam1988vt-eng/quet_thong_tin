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
    st.session_state.detected_fields = [] # lưu tạm các trường phát hiện từ ảnh

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

def ocr_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    result = reader.readtext(image, detail=0)
    return "\n".join(result)

def auto_detect_fields(text):
    """
    Tìm các dòng có dạng 'Tên trường: giá trị' hoặc 'Tên trường  giá trị'
    Trả về list các dict {name, keyword}
    """
    lines = text.split('\n')
    detected = []
    # Mẫu: tìm dòng có dấu hai chấm hoặc khoảng trắng sau tên trường
    pattern = re.compile(r'^(.+?)\s*[:：]\s*(.+)$', re.UNICODE)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            field_name = match.group(1).strip()
            value = match.group(2).strip()
            # Chỉ lấy nếu field_name không quá dài và không chứa ký tự đặc biệt
            if len(field_name) < 50 and not re.search(r'[^\w\sÀ-ỹ]', field_name):
                detected.append({
                    'name': field_name,
                    'keyword': field_name + ':',
                    'sample_value': value[:50]  # để hiển thị mẫu
                })
    # Loại bỏ trùng lặp theo tên
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
                # thử không có dấu hai chấm
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

# Sidebar: Quản lý cấu hình trường (thủ công)
with st.sidebar:
    st.header("⚙️ Quản lý trường thông tin")
    
    # Nếu có trường phát hiện tạm, hiển thị để chọn
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
            # Lưu lựa chọn
            detected_choices.append({
                'original': d,
                'keep': keep,
                'new_name': new_name,
                'keyword': new_name + ':'   # tự động dùng tên mới làm keyword
            })
        if st.button("💾 Lưu các trường đã chọn vào cấu hình"):
            for choice in detected_choices:
                if choice['keep']:
                    add_field(choice['new_name'], choice['keyword'], keep=True)
            st.session_state.detected_fields = []  # xóa tạm
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
                    st.info("Không có trường nào để trích xuất (có thể các trường đều bị tắt 'Giữ').")
            else:
                # Nếu chỉ có detected_fields nhưng chưa lưu, hướng dẫn qua sidebar
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
    **1. Tự động phát hiện trường (khi chưa có trường nào):**
    - Chụp hoặc tải ảnh lên, hệ thống sẽ OCR và tìm các dòng có dấu hai chấm.
    - Sau khi phát hiện, vào **sidebar** (bên trái), bạn sẽ thấy danh sách các trường vừa tìm thấy.
    - Tick chọn những trường bạn muốn giữ, có thể sửa tên trường cho đúng.
    - Nhấn **"Lưu các trường đã chọn"** để đưa vào cấu hình.

    **2. Trích xuất thông tin:**
    - Từ lần chụp thứ 2 trở đi, hệ thống sẽ tự động áp dụng các trường đã chọn để lấy giá trị.
    - Bạn có thể chỉnh sửa giá trị trước khi thêm vào bảng.

    **3. Quản lý trường thủ công:**
    - Bạn có thể thêm, sửa, xóa, bật/tắt trường bất cứ lúc nào trong sidebar.

    **4. Xuất Excel và chia sẻ qua Zalo:**
    - Vào tab "Xuất Excel", nhấn nút tạo file, tải về máy và gửi qua Zalo.
    """)
