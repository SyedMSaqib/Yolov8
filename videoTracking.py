from ultralytics import YOLO


model = YOLO("yolov8n.pt")  


results = model.track(source=0, save=True, show=True)


