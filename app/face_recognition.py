import numpy as np
import sklearn
import pickle
import cv2


# Load all models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml') # cascade classifier
model_svm =  pickle.load(open('./model/model_svm.pickle',mode='rb')) # machine learning model (SVM)
pca_models = pickle.load(open('./model/pca_dict.pickle',mode='rb')) # pca dictionary
model_pca = pca_models['pca'] # PCA model
mean_face_arr = pca_models['mean_face'] # Mean Face


def faceRecognitionPipeline(filename, path=True):
    if path:
        img = cv2.imread(filename)  # BGR
    else:
        img = filename  # array
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.5, 5)
    predictions = []
    
    for x, y, w, h in faces:
        roi = gray[y:y+h, x:x+w]
        roi = roi / 255.0
        roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA if roi.shape[1] > 100 else cv2.INTER_CUBIC)
        roi_reshape = roi_resize.reshape(1, 10000)
        roi_mean = roi_reshape - mean_face_arr
        eigen_image = model_pca.transform(roi_mean)
        eig_img = model_pca.inverse_transform(eigen_image)
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image).max()
        
        text = "%s : %d" % (results[0], prob_score * 100)
        color = (255, 255, 0) if results[0] == 'male' else (255, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)
        
        predictions.append({'roi': roi, 'eig_img': eig_img, 'prediction_name': results[0], 'score': prob_score})
    
    return img, predictions

def processVideo(video_source=0, output_path='./static/predict/processed_video.avi'):
    cap = cv2.VideoCapture(video_source)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change codec to 'MP4V' for .mp4 output
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process each frame through face recognition pipeline
        pred_img, _ = faceRecognitionPipeline(frame, path=False)
        
        # Write the processed frame to the output video
        out.write(pred_img)
        
        # Display the frame for live preview (optional)
        cv2.imshow('Processing Video(PRESS Q TO QUIT THE WINDOW)', pred_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return out

