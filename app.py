import streamlit as st
import pandas as pd
import easyocr
import re
from io import BytesIO
from PIL import Image
import numpy as np

# --------------------- CÀI ĐẶT BAN ĐẦU ---------------------
st.set_page_config(page_title="OCR Info Extractor", layout="wide")

# Khởi tạo EasyOCR (chạy trên CPU để tiết kiệm tài nguyên)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['vi', 'en'], gpu=False)

reader = load_ocr()

# --------------------- KHỞI TẠO SESSION STATE ---------------------
if 'fields' not in st.session_state:
    st.session_state.fields = []      # danh sách các trường: [{'id': int, 'name': str, 'keyword': str, 'keep': bool}]
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'field_counter' not in st.session_state:
    st.session_state.field_counter = 0

# --------------------- HÀM TIỆN ÍCH ---------------------
def add_field(name, keyword, keep=True):
    """Thêm một trường mới với id tự tăng"""
    st.session_state.field_counter += 1
    st.session_state.fields.append({
        'id': st.session_state.field_counter,
        'name': name,
        'keyword': keyword,
        'keep': keep
    })

def remove_field(field_id):
    """Xóa trường theo id"""
    st.session_state.fields = [f for f in st.session_state.fields if f['id'] != field_id]

def update_field(field_id, name=None, keyword=None, keep=None):
    """Cập nhật thông tin của trường"""
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
    """Nhận diện chữ từ ảnh (PIL Image hoặc numpy array)"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    result = reader.readtext(image, detail=0)
    return "\n".join(result)

def extract_values(text, fields):
    """Trích xuất giá trị cho từng trường dựa trên từ khóa"""
    row = {}
    for f in fields:
        if f['keep']:
            pattern = re.compile(rf"{re.escape(f['keyword'])}\s*(.+)", re.IGNORECASE | re.UNICODE)
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
            else:
                # Thử tìm kiếm không có dấu hai chấm
                pattern2 = re.compile(rf"{re.escape(f['keyword'])}\s+(.+)", re.IGNORECASE | re.UNICODE)
                match2 = pattern2.search(text)
                if match2:
                    value = match2.group(1).strip()
                else:
                    value = ""
            row[f['name']] = value
    return row

# --------------------- GIAO DIỆN CHÍNH ---------------------
st.title("📄 Trích xuất thông tin từ ảnh chụp")
st.markdown("---")

# Sidebar: Quản lý cấu hình trường
with st.sidebar:
    st.header("⚙️ Quản lý trường thông tin")
    
    # Thêm trường mới
    with st.expander("➕ Thêm trường mới", expanded=False):
        new_name = st.text_input("Tên trường (ví dụ: Họ và tên)")
        new_keyword = st.text_input("Từ khóa tìm kiếm (ví dụ: 'Họ tên:' hoặc 'Họ và tên:')")
        if st.button("Thêm trường"):
            if new_name and new_keyword:
                add_field(new_name, new_keyword)
                st.success(f"Đã thêm trường '{new_name}'")
                st.experimental_rerun()
            else:
                st.error("Vui lòng nhập đầy đủ tên và từ khóa")
    
    # Hiển thị danh sách trường hiện tại
    st.subheader("📋 Danh sách trường")
    if not st.session_state.fields:
        st.info("Chưa có trường nào. Hãy thêm trường ở trên.")
    else:
        for idx, f in enumerate(st.session_state.fields):
            with st.container():
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    # Cho phép sửa tên trường
                    new_name = st.text_input(f"Tên #{f['id']}", value=f['name'], key=f"name_{f['id']}")
                with col2:
                    new_keyword = st.text_input(f"Từ khóa #{f['id']}", value=f['keyword'], key=f"kw_{f['id']}")
                with col3:
                    keep = st.checkbox("Giữ", value=f['keep'], key=f"keep_{f['id']}")
                # Nút cập nhật và xóa
                col4, col5 = st.columns([1, 1])
                with col4:
                    if st.button("Cập nhật", key=f"update_{f['id']}"):
                        update_field(f['id'], name=new_name, keyword=new_keyword, keep=keep)
                        st.success("Đã cập nhật")
                        st.experimental_rerun()
                with col5:
                    if st.button("🗑️ Xóa", key=f"del_{f['id']}"):
                        remove_field(f['id'])
                        st.warning("Đã xóa trường")
                        st.experimental_rerun()
                st.markdown("---")
    
    # Nút xóa tất cả trường
    if st.button("🗑️ Xóa tất cả trường"):
        st.session_state.fields = []
        st.session_state.field_counter = 0
        st.experimental_rerun()
    
    st.markdown("---")
    st.caption("Lưu ý: Các trường không được 'Giữ' sẽ bị bỏ qua khi trích xuất.")

# Main area
tab1, tab2, tab3 = st.tabs(["📸 Chụp ảnh / Upload", "📊 Bảng dữ liệu", "📥 Xuất Excel"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        # Cho phép chụp ảnh từ camera hoặc upload
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
        # OCR
        with st.spinner("Đang nhận diện chữ..."):
            text = ocr_image(image)
        st.text_area("📝 Văn bản nhận diện", text, height=150)

        # Nếu có trường, tiến hành trích xuất
        if st.session_state.fields:
            # Trích xuất giá trị
            row = extract_values(text, st.session_state.fields)
            
            # Hiển thị và cho phép chỉnh sửa
            st.subheader("🔎 Thông tin trích xuất")
            edited_row = {}
            cols = st.columns(min(len(row), 4))
            for i, (key, val) in enumerate(row.items()):
                with cols[i % len(cols)]:
                    edited_row[key] = st.text_input(f"{key}", value=val, key=f"edit_{key}_{i}")
            
            if st.button("➕ Thêm vào bảng", use_container_width=True):
                if not edited_row:
                    st.warning("Không có dữ liệu nào để thêm")
                else:
                    new_df = pd.DataFrame([edited_row])
                    st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
                    st.success("✅ Đã thêm dữ liệu vào bảng!")
                    st.balloons()
                    st.experimental_rerun()
        else:
            st.warning("⚠️ Bạn chưa thiết lập trường thông tin. Vào sidebar để thêm trường.")

with tab2:
    st.subheader("📊 Bảng dữ liệu đã thu thập")
    if not st.session_state.data.empty:
        st.dataframe(st.session_state.data, use_container_width=True)
        # Thống kê
        st.caption(f"Tổng số dòng: {len(st.session_state.data)}")
    else:
        st.info("Chưa có dữ liệu. Hãy thêm ảnh và trích xuất thông tin.")

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

# --------------------- RESET TOÀN BỘ ---------------------
if st.sidebar.button("🔄 Reset toàn bộ dữ liệu"):
    st.session_state.clear()
    st.experimental_rerun()

# --------------------- HƯỚNG DẪN SỬ DỤNG ---------------------
with st.expander("📖 Hướng dẫn sử dụng"):
    st.markdown("""
    **1. Thiết lập trường thông tin** (bên sidebar):
    - Nhập tên trường (ví dụ: Họ và tên)
    - Nhập từ khóa tìm kiếm (ví dụ: 'Họ tên:' hoặc 'Họ và tên:')
    - Nhấn "Thêm trường"
    - Có thể bật/tắt, sửa, xóa trường

    **2. Trích xuất dữ liệu**:
    - Chụp ảnh hoặc tải ảnh lên
    - Ứng dụng sẽ tự động nhận diện chữ và trích xuất thông tin theo các trường đã thiết lập
    - Kiểm tra và sửa nếu cần, sau đó nhấn "Thêm vào bảng"

    **3. Xem bảng và xuất Excel**:
    - Dữ liệu tích lũy sẽ hiển thị ở tab "Bảng dữ liệu"
    - Ở tab "Xuất Excel", nhấn nút để tải file .xlsx về máy và chia sẻ qua Zalo
    """)
