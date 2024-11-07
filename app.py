
# Author: vlarobbyk
# Modified by: Samuel Pardo y Jairo Salazar
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.


from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def apply_morphological_operations(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    kernel_sizes = [21, 37, 55]
    results = {}

    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # Erosión
        erosion = cv2.erode(image, kernel)
        results[f'erosion_{kernel_size}'] = erosion

        # Dilatación
        dilation = cv2.dilate(image, kernel)
        results[f'dilation_{kernel_size}'] = dilation

        # Top Hat
        top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        results[f'top_hat_{kernel_size}'] = top_hat

        # Black Hat
        black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        results[f'black_hat_{kernel_size}'] = black_hat

        # Imagen Original + (Top Hat - Black Hat)
        enhanced_image = cv2.add(image, cv2.subtract(top_hat, black_hat))
        results[f'imagen_mejorada_{kernel_size}'] = enhanced_image

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('files[]')
    all_results = {}

    for file in files:
        if file.filename == '':
            return redirect(request.url)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        results = apply_morphological_operations(file_path)

        all_results[file.filename] = {'original': file.filename}
        for key, img in results.items():
            result_filename = f"{key}_{file.filename}"
            all_results[file.filename][key] = result_filename
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, result_filename), img)

    return render_template('results.html', results=all_results)

if __name__ == '__main__':
    app.run(debug=True)

