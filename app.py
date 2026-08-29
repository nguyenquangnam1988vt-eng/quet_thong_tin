import streamlit as st
import pandas as pd
import re
from io import BytesIO
from PIL import Image, ImageEnhance
import numpy as np

# ---- Bắt lỗi import PaddleOCR ----
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError as e:
    PADDLE_AVAILABLE = False
    PADDLE_ERROR = str(e)

# --------------------- CÀI ĐẶT BAN ĐẦU ---------------------
st.set_page_config(page_title="OCR Info Extractor (PaddleOCR)", page_icon="📄", layout="wide")

# ---- Kiểm tra và thông báo nếu thiếu thư viện ----
if not PADDLE_AVAILABLE:
    st.error(f"⚠️ Thiếu thư viện PaddleOCR hoặc phụ thuộc.\nChi tiết: {PADDLE_ERROR}\n\nVui lòng cài đặt bằng lệnh:\n"
             "```bash\npip install paddlepaddle paddleocr opencv-python-headless\n```")
    st.stop()

@st.cache_resource
def load_paddle_ocr():
    # Tắt log để gọn gàng
    return PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)

# --------------------- KHỞI TẠO SESSION STATE ---------------------
def init_session_state():
    if 'fields' not in st.session_state:
        st.session_state.fields = []
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame()
    if 'field_counter' not in st.session_state:
        st.session_state.field_counter = 0
    if 'detected_fields' not in st.session_state:
        st.session_state.detected_fields = []
    if 'last_text' not in st.session_state:
        st.session_state.last_text = ""

init_session_state()

# --------------------- HÀM TIỀN XỬ LÝ ẢNH ---------------------
def preprocess_image(image_input):
    try:
        if hasattr(image_input, 'getvalue'):
            img = Image.open(BytesIO(image_input.getvalue()))
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            img = Image.open(BytesIO(image_input))
            
        # Chuyển sang grayscale
        if img.mode != 'L':
            img = img.convert('L')
            
        # Tăng độ tương phản
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # Resize nếu ảnh quá lớn
        max_size = 1500
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        return np.array(img)
    except Exception as e:
        st.error(f"Lỗi tiền xử lý ảnh: {e}")
        return None

# --------------------- HÀM OCR ---------------------
def ocr_image(image_input):
    img_np = preprocess_image(image_input)
    if img_np is None:
        return ""
    try:
        ocr = load_paddle_ocr()
        result = ocr.ocr(img_np, cls=True)
        
        # Kiểm tra kết quả rỗng
        if not result or not result[0]:
            return ""
            
        lines = []
        for line in result:
            for word_info in line:
                # word_info: [bbox, (text, confidence)]
                lines.append(word_info[1][0])
        return "\n".join(lines)
    except Exception as e:
        st.error(f"Lỗi OCR: {e}")
        return ""

# --------------------- PHÁT HIỆN TRƯỜNG ---------------------
def auto_detect_fields(text):
    if not text.strip():
        return []
    
    detected = []
    pattern = re.compile(r'([^:：\n]+?)\s*[:：]\s*([^\n]+)', re.UNICODE)
    matches = pattern.findall(text)
    
    seen = set()
    for field_name, value in matches:
        field_name = field_name.strip()
        value = value.strip()
        if 0 < len(field_name) < 50 and not re.search(r'[^\w\sÀ-ỹ]', field_name) and len(value) > 0:
            if field_name not in seen:
                seen.add(field_name)
                detected.append({
                    'name': field_name,
                    'keyword': field_name + ':',
                    'sample_value': value[:50]
                })
    return detected

# --------------------- QUẢN LÝ TRƯỜNG ---------------------
def add_field(name, keyword, keep=True):
    st.session_state.field_counter += 1
    st.session_state.fields.append({
        'id': st.session_state.field_counter,
        'name': name.strip(),
        'keyword': keyword.strip(),
        'keep': keep
    })

def remove_field(field_id):
    st.session_state.fields = [f for f in st.session_state.fields if f['id'] != field_id]

def update_field(field_id, name=None, keyword=None, keep=None):
    for f in st.session_state.fields:
        if f['id'] == field_id:
            if name is not None: f['name'] = name.strip()
            if keyword is not None: f['keyword'] = keyword.strip()
            if keep is not None: f['keep'] = keep
            break

def extract_values(text, fields):
    row = {}
    for f in fields:
        if f['keep']:
            pattern = re.compile(rf"{re.escape(f['keyword'])}\s*(.+)", re.IGNORECASE | re.UNICODE)
            match = pattern.search(text)
            if not match:
                # Dự phòng nếu OCR sót dấu hai chấm
                kw_no_colon = f['keyword'].replace(':', '').strip()
                pattern2 = re.compile(rf"{re.escape(kw_no_colon)}\s+(.+)", re.IGNORECASE | re.UNICODE)
                match = pattern2.search(text)
            row[f['name']] = match.group(1).strip() if match else ""
    return row

# --------------------- GIAO DIỆN CHÍNH ---------------------
st.title("📄 Trích xuất thông tin với PaddleOCR")
st.markdown("---")

# --------------------- SIDEBAR ---------------------
with st.sidebar:
    st.header("⚙️ Quản lý trường thông tin")
    
    # 1. Trường phát hiện tự động
    if st.session_state.detected_fields:
        st.subheader("🔍 Các trường vừa phát hiện")
        st.info("Chọn các trường muốn lưu lại.")
        detected_choices = []
        for idx, d in enumerate(st.session_state.detected_fields):
            col1, col2 = st.columns([1, 3])
            with col1:
                keep = st.checkbox("Giữ", value=True, key=f"detect_keep_{idx}")
            with col2:
                new_name = st.text_input("Tên trường", value=d['name'], key=f"detect_name_{idx}", label_visibility="collapsed")
            detected_choices.append({
                'keep': keep,
                'new_name': new_name,
                'keyword': new_name + ':'
            })
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("💾 Lưu đã chọn", use_container_width=True):
                count = 0
                for choice in detected_choices:
                    if choice['keep']:
                        add_field(choice['new_name'], choice['keyword'], keep=True)
                        count += 1
                st.session_state.detected_fields = []
                st.success(f"Đã thêm {count} trường!")
                st.experimental_rerun()
        with col_btn2:
            if st.button("🗑️ Bỏ qua", use_container_width=True):
                st.session_state.detected_fields = []
                st.experimental_rerun()
        st.markdown("---")
    
    # 2. Thêm trường thủ công
    with st.expander("➕ Thêm trường thủ công", expanded=False):
        new_name = st.text_input("Tên trường", placeholder="Ví dụ: Họ tên")
        new_keyword = st.text_input("Từ khóa tìm kiếm", placeholder="Ví dụ: Họ tên:")
        if st.button("Thêm trường", use_container_width=True):
            if new_name and new_keyword:
                add_field(new_name, new_keyword)
                st.success(f"Đã thêm trường '{new_name}'")
                st.experimental_rerun()
            else:
                st.error("Vui lòng nhập đủ tên và từ khóa.")
    
    # 3. Danh sách trường đã lưu
    st.subheader("📋 Danh sách trường đã lưu")
    if not st.session_state.fields:
        st.info("Chưa có trường nào được cấu hình.")
    else:
        for f in st.session_state.fields:
            with st.container(border=True):
                new_name = st.text_input(f"Tên #{f['id']}", value=f['name'], key=f"name_{f['id']}")
                new_keyword = st.text_input(f"Từ khóa #{f['id']}", value=f['keyword'], key=f"kw_{f['id']}")
                keep = st.checkbox("Trích xuất trường này", value=f['keep'], key=f"keep_{f['id']}")
                
                col4, col5 = st.columns([1, 1])
                with col4:
                    if st.button("Cập nhật", key=f"update_{f['id']}", use_container_width=True):
                        update_field(f['id'], name=new_name, keyword=new_keyword, keep=keep)
                        st.success("Đã cập nhật")
                with col5:
                    if st.button("Xóa", key=f"del_{f['id']}", use_container_width=True):
                        remove_field(f['id'])
                        st.experimental_rerun()

        if st.button("🗑️ Xóa tất cả trường", use_container_width=True):
            st.session_state.fields = []
            st.session_state.field_counter = 0
            st.experimental_rerun()
    
    st.markdown("---")
    if st.button("🔄 Reset toàn bộ ứng dụng", use_container_width=True):
        st.session_state.clear()
        st.experimental_rerun()

# --------------------- VÙNG LÀM VIỆC CHÍNH ---------------------
tab1, tab2, tab3 = st.tabs(["📸 Chụp ảnh / Upload", "📊 Bảng dữ liệu", "📥 Xuất Excel"])

with tab1:
    col_img, col_res = st.columns([1, 1], gap="large")
    
    with col_img:
        option = st.radio("Chọn nguồn ảnh:", ("📁 Tải ảnh lên", "📷 Chụp từ camera"), horizontal=True)
        image = None
        
        if option == "📷 Chụp từ camera":
            st.caption("💡 Trên điện thoại, có thể chuyển sang camera sau bằng nút đổi camera.")
            image = st.camera_input("Chụp tài liệu")
        else:
            uploaded = st.file_uploader("Chọn ảnh (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded)
                st.image(image, caption="Ảnh đầu vào", use_container_width=True)

        if image is not None:
            if st.button("🔍 Tiến hành trích xuất (OCR)", type="primary", use_container_width=True):
                with st.spinner("Đang quét ảnh với PaddleOCR..."):
                    text = ocr_image(image)
                    st.session_state.last_text = text
                    
                    if not st.session_state.fields and not st.session_state.detected_fields:
                        detected = auto_detect_fields(text)
                        if detected:
                            st.session_state.detected_fields = detected
                    st.experimental_rerun()
    
    with col_res:
        if st.session_state.last_text:
            text = st.session_state.last_text
            
            with st.expander("📝 Xem toàn bộ văn bản gốc", expanded=False):
                st.text_area(label="Văn bản nhận diện", value=text, height=200, label_visibility="collapsed")
            
            if st.session_state.detected_fields:
                st.info("👈 Đã phát hiện các trường dữ liệu mới. Vui lòng kiểm tra Sidebar để lưu lại.")
                
            if st.session_state.fields:
                st.subheader("🔎 Kết quả trích xuất")
                row = extract_values(text, st.session_state.fields)
                
                if any(row.values()):
                    edited_row = {}
                    for key, val in row.items():
                        edited_row[key] = st.text_input(f"{key}", value=val)
                        
                    if st.button("➕ Thêm vào bảng dữ liệu", type="primary"):
                        new_df = pd.DataFrame([edited_row])
                        if st.session_state.data.empty:
                            st.session_state.data = new_df
                        else:
                            st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
                        st.success("✅ Đã lưu dữ liệu vào bảng!")
                        st.balloons()
                else:
                    st.warning("Không trích xuất được giá trị nào. Hãy thử kiểm tra lại 'Từ khóa' trong cấu hình trường ở Sidebar.")
            else:
                if not st.session_state.detected_fields:
                    st.warning("Bạn chưa cấu hình Trường dữ liệu nào. Vui lòng thêm thủ công bên Sidebar.")

with tab2:
    st.subheader("📊 Bảng dữ liệu đã thu thập")
    if not st.session_state.data.empty:
        edited_df = st.data_editor(st.session_state.data, use_container_width=True, num_rows="dynamic")
        st.session_state.data = edited_df
        st.caption(f"Tổng số dòng: {len(st.session_state.data)}")
    else:
        st.info("Chưa có dữ liệu nào được thu thập.")

with tab3:
    st.subheader("📥 Xuất dữ liệu")
    if st.session_state.data.empty:
        st.warning("Bảng dữ liệu đang trống, chưa thể xuất Excel!")
    else:
        st.write("Tải xuống bảng dữ liệu hiện tại dưới định dạng `.xlsx`")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.data.to_excel(writer, index=False, sheet_name='Data')
        
        st.download_button(
            label="⬇️ Tải file Excel",
            data=output.getvalue(),
            file_name="du_lieu_trich_xuat.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

with st.expander("📖 Hướng dẫn sử dụng"):
    st.markdown("""
    **PaddleOCR** cho độ chính xác rất cao với Tiếng Việt. 
    1. **Tải ảnh/Chụp ảnh** và nhấn nút **Tiến hành trích xuất**.
    2. Nếu đây là lần đầu tiên chạy, công cụ sẽ cố gắng **Tự động phát hiện trường** (dựa vào dấu hai chấm `:`). Hãy chọn và lưu các trường này ở thanh điều hướng bên trái (Sidebar).
    3. Nếu kết quả OCR bị sai sót cấu trúc, bạn có thể thiết lập **Tên trường** và **Từ khóa** thủ công.
    4. Sửa lại dữ liệu kết quả trực tiếp trên khung giao diện nếu máy nhận diện sai chữ.
    5. Nhấn **Thêm vào bảng dữ liệu** để lưu lại thông tin. Bảng có thể chỉnh sửa trực tiếp ở Tab 2.
    """)
