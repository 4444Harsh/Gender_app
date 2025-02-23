import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from app.face_recognition import processVideo
import matplotlib.image as matimg
from flask import render_template, request, Response

UPLOAD_FOLDER = 'static/upload'
video_capture = None  # Global variable to manage video feed

def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        
        # Process the uploaded image
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)
        
        # Generate report
        report = []
        for i, obj in enumerate(predictions):
            gray_image = obj['roi']
            eigen_image = obj['eig_img'].reshape(100, 100)
            gender_name = obj['prediction_name']
            score = round(obj['score'] * 100, 2)
            
            # Save grayscale and eigen images
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}', gray_image, cmap='gray')
            matimg.imsave(f'./static/predict/{eig_image_name}', eigen_image, cmap='gray')
            
            report.append([gray_image_name, eig_image_name, gender_name, score])
        
        return render_template('gender.html', fileupload=True, report=report)
    
    return render_template('gender.html', fileupload=False)

def video_upload():
    if request.method == 'POST':
        if 'video_name' not in request.files:
            return "No file part", 400  

        f = request.files['video_name']
        if f.filename == '':
            return "No selected file", 400  

        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)  

        # Process the uploaded video
        output_video_path = os.path.join('static/predict', 'processed_video.avi')  # Ensure relative path
        processVideo(path, output_video_path)  # Call processing function

        # Pass the processed video path to the template
        return render_template('video_result.html', processed_video=output_video_path)

    return render_template('gender.html')




def video_feed():
    global video_capture
    video_capture = cv2.VideoCapture(0)  # Turn on the camera

    def generate_frames():
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            pred_img, _ = faceRecognitionPipeline(frame, path=False)
            _, buffer = cv2.imencode('.jpg', pred_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def stop_video_feed():
    global video_capture
    if video_capture:
        video_capture.release()  # Release the camera
        video_capture = None
    return "Video feed stopped!"

