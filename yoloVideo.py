from ultralytics import YOLO


model = YOLO("yolov8n.pt")  


results = model.predict(source="https://www.youtube.com/watch?v=TI4zcmScFQk",show=True, save=True, classes=[0])


