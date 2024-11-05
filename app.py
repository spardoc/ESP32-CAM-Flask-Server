
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

# Ruta para almacenar las imágenes subidas
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def apply_morphological_operations(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Definir el tamaño de la máscara
    kernel_size = 37
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Erosión
    erosion = cv2.erode(image, kernel)

    # Dilatación
    dilation = cv2.dilate(image, kernel)

    # Top Hat
    top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    # Black Hat
    black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    # Imagen Original + (Top Hat - Black Hat)
    enhanced_image = cv2.add(image, cv2.subtract(top_hat, black_hat))

    return erosion, dilation, top_hat, black_hat, enhanced_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('files[]')
    results = {}

    for file in files:
        if file.filename == '':
            return redirect(request.url)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        erosion, dilation, top_hat, black_hat, enhanced_image = apply_morphological_operations(file_path)

        # Guardar las imágenes procesadas
        results[file.filename] = {
            'erosion': 'erosion_' + file.filename,
            'dilation': 'dilation_' + file.filename,
            'top_hat': 'top_hat_' + file.filename,
            'black_hat': 'black_hat_' + file.filename,
            'enhanced_image': 'enhanced_image_' + file.filename
        }
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, results[file.filename]['erosion']), erosion)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, results[file.filename]['dilation']), dilation)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, results[file.filename]['top_hat']), top_hat)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, results[file.filename]['black_hat']), black_hat)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, results[file.filename]['enhanced_image']), enhanced_image)

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
