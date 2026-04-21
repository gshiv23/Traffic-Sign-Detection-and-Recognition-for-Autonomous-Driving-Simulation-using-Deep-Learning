from ultralytics import YOLO

class YOLODetector:

    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model(frame)

        boxes = []

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))

        return boxes