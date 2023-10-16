from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
  data = 'D:/vscode/vsdaima/python/YOLOv8/yolov8/datasets/demo/data.yaml',
  epochs = 10,
  workers = 0,
  imgsz = 640
)
