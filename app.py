import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
import datetime
from io import BytesIO
import zipfile
from streamlit_javascript import st_javascript

def process_image(image, blur_value, canny_min, canny_max, kernel_size):
    """クラック検出処理"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # ガウシアンブラーの適用（奇数にする）
    if blur_value % 2 == 0:
        blur_value += 1
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    
    # Cannyエッジ検出
    edges = cv2.Canny(blurred, canny_min, canny_max)
    
    # 形態学的処理
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 結果の結合
    morphed_color = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
    
    return morphed_color

def create_zip(processed_images, parameters_text, image_names, timestamp):
    """処理後画像とパラメータを ZIP ファイルにまとめる"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr(f"parameters_{timestamp}.txt", parameters_text.strip())
        
        for img_name, image_array in zip(image_names, processed_images):
            img_buffer = BytesIO()
            processed_file_name = os.path.splitext(img_name)[0] + "_processed.png"
            Image.fromarray(image_array).save(img_buffer, format="PNG")
            zip_file.writestr(processed_file_name, img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def update_slider(var_name):
    st.session_state[f"{var_name}_slider"] = st.session_state[f"{var_name}_input"]

def update_input(var_name):
    st.session_state[f"{var_name}_input"] = st.session_state[f"{var_name}_slider"]

st.set_page_config(layout="wide")
st.title("クラック検出アプリ")

uploaded_files = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
window_width = st_javascript("window.innerWidth")

if uploaded_files:
    tab_names = [uploaded_file.name for uploaded_file in uploaded_files]
    processed_images = []
    image_names = []
    
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("パラメータ調整")
            for var, default, min_val, max_val, step in [
                ("blur", 1, 1, 20, 2),
                ("canny_min", 50, 0, 500, 1),
                ("canny_max", 150, 0, 500, 1),
                ("kernel", 1, 1, 20, 2)
            ]:
                if f"{var}_input" not in st.session_state:
                    st.session_state[f"{var}_input"] = default
                    st.session_state[f"{var}_slider"] = default
                st.number_input(var.replace("_", " ").title(), min_value=min_val, max_value=max_val, step=step,
                                key=f"{var}_input", on_change=lambda v=var: update_slider(v))
                st.slider(var.replace("_", " ").title(), min_val, max_val, step=step, key=f"{var}_slider",
                          on_change=lambda v=var: update_input(v))
        
        with col2:
            st.subheader("画像表示")
            tabs = st.tabs(tab_names)
            for tab, uploaded_file in zip(tabs, uploaded_files):
                with tab:
                    image = Image.open(uploaded_file)
                    image = np.array(image)
                    
                    # RGBAをRGBに変換（必要な場合）
                    if image.shape[-1] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    
                    processed_image = process_image(image, st.session_state.blur_input, 
                                                    st.session_state.canny_min_input, 
                                                    st.session_state.canny_max_input, 
                                                    st.session_state.kernel_input)
            

                    # 画像の結合
                    # h, w, _ = image.shape
                    # if w > h:
                    #     combined_image = np.vstack((image, processed_image))
                    # else:
                    #     combined_image = np.hstack((image, processed_image))
                    combined_image = np.hstack((image, processed_image))

                    st.image(combined_image, caption=f"処理後画像: {uploaded_file.name}", use_container_width=True)
                    processed_images.append(processed_image)
                    image_names.append(uploaded_file.name)


        with col1:
            st.subheader("保存・ダウンロード")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            parameters_text = f"""
GaussianBlur: {st.session_state.blur_input}
Canny Min: {st.session_state.canny_min_input}
Canny Max: {st.session_state.canny_max_input}
Kernel Size: {st.session_state.kernel_input}
Original Image Path: {', '.join(image_names)}""".strip()
            zip_buffer = create_zip(processed_images, parameters_text, image_names, timestamp)
            st.download_button("処理後画像とパラメータをダウンロード (ZIP)", data=zip_buffer, file_name=f"processed_data_{timestamp}.zip", mime="application/zip")
