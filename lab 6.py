# Step 1: Install Required Libraries
# Run these commands in your terminal before executing the script:
# pip install ultralytics roboflow torch torchvision opencv-python exifread folium fastapi uvicorn
import os
import cv2
import exifread
import smtplib
from email.mime.text import MIMEText
import folium
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import shutil
from roboflow import Roboflow

# Step 2: Download Dataset from Roboflow
def download_roboflow_dataset():
    # Read the API key from the environment variable
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Roboflow API key not found in environment variables. Please set the ROBOFLOW_API_KEY environment variable.")

    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)

    # Download dataset (replace with your dataset details)
    project = rf.workspace("cattle-herd").project("cattle-herd-detection")
    dataset = project.version(1).download("yolov8")

    print(f"Dataset downloaded to {dataset.location}")

# Step 3: Train YOLOv8 Model
def train_yolo_model():
    # Load pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(data="cattle-herd-detection-1/data.yaml", epochs=100, imgsz=640, batch=16, name="herd_detection")

    print("Training complete. Model saved in 'runs/detect/herd_detection/weights/best.pt'")

# Step 4: Geolocation Tagging
def get_gps(image_path):
    """Extract GPS coordinates from image metadata."""
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
        lat = tags.get('GPS GPSLatitude')
        lon = tags.get('GPS GPSLongitude')
    return lat, lon

# Step 5: Alert System
def send_alert(lat, lon):
    """Send an email alert with GPS coordinates."""
    sender = "your_email@gmail.com"
    receiver = "receiver@example.com"
    password = "your_app_password"  # Use App Password for Gmail

    # Create message
    message = MIMEText(f"Herd detected at Latitude: {lat}, Longitude: {lon}")
    message["Subject"] = "Herd Detection Alert"
    message["From"] = sender
    message["To"] = receiver

    # Send email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, message.as_string())
    print(f"Alert sent: Herd detected at {lat}, {lon}")

# Step 6: Map Visualization
def plot_on_map(lat, lon):
    """Plot herd location on a map using Folium."""
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], popup="Herd Detected").add_to(m)
    m.save("herd_map.html")
    print("Map saved as 'herd_map.html'")

# Step 7: Full Integration
def detect_herds_in_video(video_path):
    """Detect herds in a video and send alerts."""
    # Load trained model
    model = YOLO("runs/detect/herd_detection/weights/best.pt")

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame (adjust as needed)
        frame_count += 1
        if frame_count % 10 != 0:
            continue

        # Detect herds
        results = model.predict(frame, conf=0.5)
        for box in results[0].boxes:
            if box.cls == 0:  # Class 'herd'
                # Get GPS from frame (or sync with drone telemetry)
                lat, lon = 37.7749, -122.4194  # Replace with actual GPS extraction logic
                send_alert(lat, lon)
                plot_on_map(lat, lon)

    cap.release()
    print("Video processing complete.")

# Step 8: FastAPI Deployment
app = FastAPI()

@app.post("/detect")
async def detect_herd(file: UploadFile = File(...)):
    """API endpoint to detect herds in an uploaded image."""
    # Save uploaded file
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load trained model
    model = YOLO("runs/detect/herd_detection/weights/best.pt")

    # Detect herds
    results = model.predict("temp.jpg")
    gps = get_gps("temp.jpg")

    return {"gps": gps, "detections": results[0].boxes}

@app.get("/map", response_class=HTMLResponse)
async def show_map(lat: float, lon: float):
    """API endpoint to display herd location on a map."""
    plot_on_map(lat, lon)
    return open("herd_map.html").read()

# Step 9: Run the Application
if __name__ == "__main__":
    # Step 1: Download Roboflow dataset
    download_roboflow_dataset()

    # Step 2: Train the model
    train_yolo_model()

    # Step 3: Detect herds in a video (uncomment to run)
    # detect_herds_in_video("drone_footage.mp4")

    # Step 4: Run FastAPI server (uncomment to deploy)
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)