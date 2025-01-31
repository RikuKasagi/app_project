import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
import datetime

def process_image(image, blur_value, canny_min, canny_max, kernel_size):
    """ひび検出処理"""
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

def save_results(original_image, processed_image, save_path, params):
    """画像とパラメータを保存"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = os.path.join(save_path, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    # 画像を保存
    original_image_path = os.path.join(save_folder, "original.png")
    processed_image_path = os.path.join(save_folder, "processed.png")
    
    original_image_pil = Image.fromarray(original_image)
    processed_image_pil = Image.fromarray(processed_image)

    original_image_pil.save(original_image_path)
    processed_image_pil.save(processed_image_path)

    # パラメータを保存
    param_path = os.path.join(save_folder, "parameters.txt")
    with open(param_path, "w") as f:
        f.write(f"GaussianBlur: {params['blur_value']}\n")
        f.write(f"Canny Min: {params['canny_min']}\n")
        f.write(f"Canny Max: {params['canny_max']}\n")
        f.write(f"Kernel Size: {params['kernel_size']}\n")
        f.write(f"Original Image Path: {params['original_path']}\n")

    return save_folder

# Streamlit UI
st.set_page_config(layout="wide")  # ワイドレイアウト
st.title("ひび検出アプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 画像の読み込み
    image = Image.open(uploaded_file)
    image = np.array(image)

    # レイアウト設定
    col1, col2 = st.columns([1, 2])  # 左にスライダー、右に画像表示

    with col1:
        st.subheader("パラメータ調整")
        
        # スライダー + 数値入力
        blur_value = st.number_input("GaussianBlur", min_value=1, max_value=20, value=1, step=2)
        blur_slider = st.slider("GaussianBlur", 1, 20, blur_value, step=2)
        if blur_slider != blur_value:
            blur_value = blur_slider

        canny_min = st.number_input("Canny Min", min_value=0, max_value=500, value=50)
        canny_min_slider = st.slider("Canny Min", 0, 500, canny_min)
        if canny_min_slider != canny_min:
            canny_min = canny_min_slider

        canny_max = st.number_input("Canny Max", min_value=0, max_value=500, value=150)
        canny_max_slider = st.slider("Canny Max", 0, 500, canny_max)
        if canny_max_slider != canny_max:
            canny_max = canny_max_slider

        kernel_size = st.number_input("Kernel Size", min_value=1, max_value=20, value=1, step=2)
        kernel_slider = st.slider("Kernel Size", 1, 20, kernel_size, step=2)
        if kernel_slider != kernel_size:
            kernel_size = kernel_slider

        # 保存先フォルダの指定
        save_path = st.text_input("保存先フォルダを入力", value=os.getcwd())

    # 画像処理（ここで処理しておく）
    processed_image = process_image(image, blur_value, canny_min, canny_max, kernel_size)

    with col1:
        # 保存ボタン（ディレクトリ設定のすぐ下に配置）
        if st.button("保存"):
            params = {
                "blur_value": blur_value,
                "canny_min": canny_min,
                "canny_max": canny_max,
                "kernel_size": kernel_size,
                "original_path": uploaded_file.name
            }

            saved_folder = save_results(image, processed_image, save_path, params)
            st.success(f"保存完了！フォルダ: {saved_folder}")

    with col2:
        st.subheader("処理後画像（元画像 + 処理後画像）")
        combined_image = np.hstack((image, processed_image))  # 画像を横並びにする
        st.image(combined_image, caption="処理後画像", use_container_width=True)
