import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
from collections import defaultdict
import numpy as np

# COCO classes (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Global variables
processed_image = None
image_path = None
video_path = None
is_video = False
tracking_mode = False

# Function to select image
def select_image():
    global image_path, is_video
    is_video = False
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if image_path:
        img = Image.open(image_path)
        img.thumbnail((1200, 1200))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img
        
        # Resize window to fit the image
        root.geometry(f"{img.width()}x{img.height() + 200}")  # Adding extra space for buttons and status bar
        status_label.config(text=f"Image loaded: {image_path}")

# Function to select video
def select_video():
    global video_path, is_video
    is_video = True
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if video_path:
        status_label.config(text=f"Video loaded: {video_path}")

# Function to perform detection or tracking (runs in a separate thread)
def process_video():
    global tracking_mode

    # Show loading indicator
    progress_bar.grid(row=5, column=0, columnspan=2, pady=10)
    loading_label.grid(row=4, column=0, columnspan=2, pady=10)
    loading_label.config(text="Processing...")
    root.update_idletasks()

    try:
        if not video_path:
            messagebox.showerror("Error", "Please select a video.")
            return

        selected_class = class_var.get()
        if selected_class not in COCO_CLASSES and selected_class != "All Objects":
            messagebox.showerror("Error", "Please select a valid class.")
            return

        class_ids = list(range(len(COCO_CLASSES))) if selected_class == "All Objects" else [COCO_CLASSES.index(selected_class)]

        if tracking_mode:
            # Tracking code
            cap = cv2.VideoCapture(video_path)
            track_history = defaultdict(lambda: [])

            resize_width = 600  
            resize_height = 1400  

            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    results = model.track(frame, persist=True)
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    annotated_frame = results[0].plot()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y))) 
                        if len(track) > 30:  
                            track.pop(0)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                    height, width = annotated_frame.shape[:2]
                    scaling_factor = min(resize_width / width, resize_height / height)
                    new_width = int(width * scaling_factor)
                    new_height = int(height * scaling_factor)
                    resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

                    cv2.imshow("YOLOv8 Tracking", resized_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()
            status_label.config(text="Video tracking complete.")
        else:
            # Detection code
            results = model.predict(source=video_path, show=True, save=True, classes=class_ids)
            status_label.config(text="Video detection complete.")
    finally:
        # Hide loading indicator
        progress_bar.grid_forget()
        loading_label.grid_forget()

# Function to perform detection on images (runs in a separate thread)
def detect_objects():
    global processed_image

    # Show loading indicator
    progress_bar.grid(row=5, column=0, columnspan=2, pady=10)
    loading_label.grid(row=4, column=0, columnspan=2, pady=10)
    loading_label.config(text="Processing...")
    root.update_idletasks()

    try:
        if not image_path:
            messagebox.showerror("Error", "Please select an image.")
            return

        selected_class = class_var.get()
        if selected_class not in COCO_CLASSES and selected_class != "All Objects":
            messagebox.showerror("Error", "Please select a valid class.")
            return

        class_ids = list(range(len(COCO_CLASSES))) if selected_class == "All Objects" else [COCO_CLASSES.index(selected_class)]

        # Perform inference
        results = model(image_path)

        # Filter results based on selected class IDs
        filtered_detections = [det for det in results[0].boxes if int(det.cls) in class_ids]

        # Load the image using OpenCV to draw bounding boxes
        image = cv2.imread(image_path)
        for det in filtered_detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            confidence = float(det.conf)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display confidence score
            label = f"{COCO_CLASSES[int(det.cls)]}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert image to RGB (from BGR) for displaying in Tkinter
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(processed_image)
        img.thumbnail((1200, 1200))
        img = ImageTk.PhotoImage(img)

        panel.configure(image=img)
        panel.image = img

        # Update status
        status_label.config(text="Detection complete.")
    finally:
        # Hide loading indicator
        progress_bar.grid_forget()
        loading_label.grid_forget()

# Function to start the detection in a separate thread
def start_detection():
    detection_thread = threading.Thread(target=detect_objects)
    detection_thread.start()

# Function to start the video processing (detection or tracking) in a separate thread
def start_video_processing():
    global tracking_mode
    tracking_mode = process_var.get() == "Tracking"
    video_thread = threading.Thread(target=process_video)
    video_thread.start()

# Function to save the processed image
def save_image():
    if processed_image is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Image Saved", "Image has been saved successfully.")
            status_label.config(text=f"Image saved: {save_path}")
    else:
        messagebox.showerror("Error", "No image to save. Please run detection first.")

# Function to toggle between image and video modes
def toggle_mode(mode):
    if mode == "Image":
        btn_select.config(command=select_image)
        btn_save_image.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)
        process_dropdown.grid_forget()
    else:
        btn_select.config(command=select_video)
        btn_save_image.grid_forget()
        process_dropdown.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)

# Main Tkinter window
root = tk.Tk()
root.title("Object Detection and Tracking with YOLOv8")
root.geometry("800x600")  # Start with a smaller window size

# Create a style for the GUI
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TCombobox", font=("Helvetica", 12))

# Layout management using grid
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

frame = ttk.Frame(root, padding="10 10 10 10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Option to select image or video
mode_var = tk.StringVar(value="Image")
image_radio = ttk.Radiobutton(frame, text="Image", variable=mode_var, value="Image", command=lambda: toggle_mode("Image"))
image_radio.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
video_radio = ttk.Radiobutton(frame, text="Video", variable=mode_var, value="Video", command=lambda: toggle_mode("Video"))
video_radio.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)

# File selection button
btn_select = ttk.Button(frame, text="Select File", command=select_image, width=20)
btn_select.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

# Detection/Tracking selection dropdown for videos
process_var = tk.StringVar(value="Detection")
process_dropdown = ttk.Combobox(frame, textvariable=process_var, values=["Detection", "Tracking"], state="readonly", width=20)
process_dropdown.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)
process_dropdown.grid_forget()  # Initially hidden

# Class selection dropdown for image and video
class_var = tk.StringVar(value="All Objects")
class_dropdown = ttk.Combobox(frame, textvariable=class_var, values=["All Objects"] + COCO_CLASSES, state="readonly", width=50)
class_dropdown.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)

# Detection/Tracking button
btn_process = ttk.Button(frame, text="Process", command=lambda: start_detection() if not is_video else start_video_processing(), width=20)
btn_process.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)

# Save Image button (visible only for images)
btn_save_image = ttk.Button(frame, text="Save Image", command=save_image, width=20)
btn_save_image.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)

# Status bar and progress bar
status_label = ttk.Label(frame, text="Select an image or video")
status_label.grid(row=4, column=0, columnspan=2, pady=10, sticky=tk.W)
progress_bar = ttk.Progressbar(frame, mode="indeterminate", length=200)
loading_label = ttk.Label(frame, text="")

# Image display panel
panel = ttk.Label(frame)
panel.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# Start Tkinter main loop
root.mainloop()
