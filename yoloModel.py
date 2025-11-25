from ultralytics import YOLO
from utils.plot_utils import plot_image_with_bboxes

class YoloModel:
    def __init__(self, args):
        self.args = args
        self.model = YOLO(args.model_path, task='detect')
    
    def predict(self, tensor):
        results=self.model.predict(tensor) 
        return results
    def extract_bboxes(self, results):
        boxes = results[0].boxes  # Assuming we have one image and accessing the first result
        boxes.conf
        predicted_bboxes = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])               # Confidence score
            cls_id = int(box.cls[0])                # Class ID
            label = self.model.names[cls_id]        # Class name from model metadata

            # Format the bbox as a dict
            predicted_bboxes.append({
                "bbox": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                "confidence": conf,
                "class_id": cls_id,
                "label": label
            })
        
        return predicted_bboxes
    def plot_bboxes(self, img_np, predicted_bboxes, output_path):
        image_with_bboxes = plot_image_with_bboxes(img_np, predicted_bboxes, save_to=f'{output_path}yolo_predicted_bboxes.png', show_plot=self.args.show_plots, tight_save=self.args.remove_all_borders_and_legends_from_images)
        return image_with_bboxes