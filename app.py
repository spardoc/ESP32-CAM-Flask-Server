
# Author: vlarobbyk
# Modified by: Samuel Pardo y Jairo Salazar
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.


from flask import Flask, render_template, Response, stream_with_context, request, jsonify

from io import BytesIO
import cv2
import numpy as np
import requests
import time

app = Flask(__name__)

# IP Address
_URL = 'http://192.168.0.239'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

# Parameters for movement detection
background = None
MAX_FRAMES = 1000
THRESH = 60
ASSIGN_VALUE = 255
ALPHA = 0.1

# Global variables for noise levels
salt_percentage = 0
pepper_percentage = 0

mask_size = 3

stream_url = ''.join([_URL, SEP, _PORT, _ST])

def update_background(current_frame, prev_bg, alpha):
    bg = alpha * current_frame + (1 - alpha) * prev_bg
    return np.uint8(bg)

def apply_clahe(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def apply_histogram_equalization(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def apply_bilateral_filter(frame):
    return cv2.bilateralFilter(frame, d=15, sigmaColor=75, sigmaSpace=75)

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy = np.copy(image)
    total_pixels = image.size

    # Salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy

def update_mask_size(val):
    global mask_size
    mask_size = MASK_SIZES[val]

def apply_filters(image):
    global mask_size
    filtered_images = []

    median_filtered = cv2.medianBlur(image, mask_size)
    blur_filtered = cv2.blur(image, (mask_size, mask_size))
    gaussian_filtered = cv2.GaussianBlur(image, (mask_size, mask_size), 0)
    filtered_images.append((median_filtered, blur_filtered, gaussian_filtered))

    return filtered_images

def apply_canny_edge_detection(frame, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def apply_sobel_edge_detection(frame, ksize=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    edges = cv2.magnitude(grad_x, grad_y)
    edges = cv2.convertScaleAbs(edges)
    return edges

def apply_laplacian_edge_detection(frame, ksize=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    edges = cv2.convertScaleAbs(edges)
    return edges

def apply_canny_with_blur(image, low_threshold=50, high_threshold=150):
    # Suavizado Gaussiano antes de aplicar Canny
    blurred_image = cv2.GaussianBlur(image, (mask_size, mask_size), 0)
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def apply_sobel_with_blur(image, ksize=3):
    # Suavizado Gaussiano antes de aplicar Sobel
    blurred_image = cv2.GaussianBlur(image, (mask_size, mask_size), 0)
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    edges = cv2.magnitude(grad_x, grad_y)
    edges = cv2.convertScaleAbs(edges)
    return edges

def apply_laplacian_with_blur(image, ksize=3):
    blurred_image = cv2.GaussianBlur(image, (mask_size, mask_size), 0)
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    edges = cv2.convertScaleAbs(edges)
    return edges

def video_capture():
    global background, salt_percentage, pepper_percentage
    res = requests.get(stream_url, stream=True)
    t = 0
    prev_time = time.time()
    
    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)

                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape

                # Detection movement
                if background is None:
                    background = gray

                diff = cv2.absdiff(background, gray)
                _, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, cv2.THRESH_BINARY)
                background = update_background(gray, background, ALPHA)
                
                # Add noise based on trackbar values
                noisy_image = add_salt_and_pepper_noise(cv_img.copy(), salt_percentage / 100, pepper_percentage / 100)
                
                # Filter Applications
                img_clahe = apply_clahe(cv_img)
                img_equalized = apply_histogram_equalization(cv_img)
                img_bilateral = apply_bilateral_filter(cv_img) 
                filtered_images = apply_filters(cv_img)
                img_canny = apply_canny_edge_detection(cv_img)
                img_sobel = apply_sobel_edge_detection(cv_img)
                img_laplacian = apply_laplacian_edge_detection(cv_img)
                img_canny_blur = apply_canny_with_blur(cv_img)
                img_sobel_blur = apply_sobel_with_blur(cv_img)
                img_laplacian_blur = apply_laplacian_with_blur(cv_img)

                # Convertir las imágenes en escala de grises a BGR
                img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
                img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2BGR)
                motion_mask = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
                img_sobel = cv2.cvtColor(img_sobel, cv2.COLOR_GRAY2BGR)
                img_laplacian = cv2.cvtColor(img_laplacian, cv2.COLOR_GRAY2BGR)
                img_canny_blur = cv2.cvtColor(img_canny_blur, cv2.COLOR_GRAY2BGR)
                img_sobel_blur = cv2.cvtColor(img_sobel_blur, cv2.COLOR_GRAY2BGR)
                img_laplacian_blur = cv2.cvtColor(img_laplacian_blur, cv2.COLOR_GRAY2BGR)
                
                # Calculate grid size for total_image
                num_rows = 5 
                total_image = np.zeros((height * num_rows, width * 3, 3), dtype=np.uint8)

                # Place base images in the grid
                total_image[:height, :width] = cv_img
                total_image[:height, width:2*width] = img_equalized
                total_image[:height, 2*width:3*width] = img_clahe
                total_image[height:2*height, :width] = img_bilateral
                total_image[height:2*height, width:2*width] = motion_mask
                total_image[height:2*height, 2*width:3*width] = noisy_image

                # Place filtered images in the grid
                row_offset = 2  # Starting from the third row
                for i, (median_filtered, blur_filtered, gaussian_filtered) in enumerate(filtered_images):
                    filters = [median_filtered, blur_filtered, gaussian_filtered]
                    for j, filtered in enumerate(filters):
                        # Resize filtered images to match height and width if necessary
                        if filtered.shape[:2] != (height, width):
                            filtered = cv2.resize(filtered, (width, height))
                        
                        # Place each filtered image in the grid
                        row = row_offset + i
                        col = j
                        total_image[row*height:(row+1)*height, col*width:(col+1)*width] = filtered
                        
                total_image[3*height:4*height, :width] = img_canny
                total_image[3*height:4*height, width:2*width] = img_sobel
                total_image[3*height:4*height, 2*width:3*width] = img_laplacian
                
                total_image[4*height:5*height, :width] = img_canny_blur
                total_image[4*height:5*height, width:2*width] = img_sobel_blur
                total_image[4*height:5*height, 2*width:3*width] = img_laplacian_blur
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time

                cv2.putText(total_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                    bytearray(encodedImage) + b'\r\n')
                
                t += 1
                if t >= MAX_FRAMES:
                    break

            except Exception as e:
                print(e)
                continue

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    
@app.route('/update_noise', methods=['POST'])
def update_noise():
    global salt_percentage, pepper_percentage
    data = request.get_json()
    salt_percentage = float(data.get('salt', 0))
    pepper_percentage = float(data.get('pepper', 0))
    print(f"Salt Noise: {salt_percentage}, Pepper Noise: {pepper_percentage}")
    return jsonify({'status': 'success'}), 200

@app.route('/update_mask_size', methods=['POST'])
def update_mask_size():
    global mask_size
    data = request.get_json()
    mask_size = int(data.get('mask_size', 3))
    print(f"Mask Size: {mask_size}")
    return jsonify({'status': 'success'}), 200

if __name__ == "__main__":
    app.run(debug=False)
