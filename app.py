import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
import datetime
from io import BytesIO
import zipfile

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

def create_zip(image_array, parameters_text, timestamp):
    """処理後画像とパラメータを ZIP ファイルにまとめる"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # 処理後画像を ZIP に追加
        img_buffer = BytesIO()
        Image.fromarray(image_array).save(img_buffer, format="PNG")
        zip_file.writestr("processed_image.png", img_buffer.getvalue())

        # パラメータを ZIP に追加
        zip_file.writestr("parameters.txt", parameters_text.strip())  # 空白削除

    zip_buffer.seek(0)  # バッファのポインタを先頭にリセット
    return zip_buffer

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

    # 画像処理
    processed_image = process_image(image, blur_value, canny_min, canny_max, kernel_size)

    with col1:
        st.subheader("保存・ダウンロード")

        # 現在の日時を取得
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # パラメータのテキストデータを作成（空白削除）
        params_text = f"""GaussianBlur: {blur_value}
Canny Min: {canny_min}
Canny Max: {canny_max}
Kernel Size: {kernel_size}
Original Image Path: {uploaded_file.name}"""

        # ZIPファイルの作成
        zip_buffer = create_zip(processed_image, params_text, timestamp)

        # ZIP ダウンロードボタン（ファイル名にタイムスタンプを付与）
        st.download_button(
            label="処理後画像とパラメータをダウンロード (ZIP)",
            data=zip_buffer,
            file_name=f"processed_data_{timestamp}.zip",
            mime="application/zip"
        )

    with col2:
        st.subheader("処理後画像（元画像 + 処理後画像）")
        combined_image = np.hstack((image, processed_image))  # 画像を横並びにする
        st.image(combined_image, caption="処理後画像", use_container_width=True)
