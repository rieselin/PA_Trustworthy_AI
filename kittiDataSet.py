import fiftyone as fo


import json
import os

from dataSet import DataSet


class KittiDataSet(DataSet):
    def __init__(self, dataset_name="kitti", split="train", max_samples=None, dataset_dir=None):
        super().__init__(dataset_name=dataset_name, split=split, max_samples=max_samples, dataset_dir=dataset_dir)
        self.dataset = self.load_dataset()
        
    def load_dataset(self):
        fo.config.dataset_zoo_dir = self.dataset_dir
        dataset = fo.zoo.load_zoo_dataset(
            self.dataset_name,
            split=self.split,
            max_samples=self.max_samples
        )
        return dataset
    def prepare_annotations(self, overwrite=False):
        # save bboxes in class_id x1 y1 x2 y1 x2 y2 x1 y2 (as text file imagename.txt)
        output_dir = self.dataset_dir +"/"+ self.dataset_name + "/" + self.split + "/"
        annotations_dir = output_dir + "annotations/"

        os.makedirs(annotations_dir, exist_ok=True)

        # check if labelfiles already exist
        if not overwrite:
            existing_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
            if len(existing_files) >= self.max_samples:
                print(f"Label files already exist in '{annotations_dir}'. Skipping save.")
                return

        for sample in self.dataset:
            # Get detections
            detections = sample.ground_truth.detections if sample.ground_truth else []

            label_lines = []
            for det in detections:
                label = det.label
                class_id = class_map.get(label, 0)

                # FiftyOne bounding boxes are [x, y, w, h] relative to image size (0–1)
                x_rel, y_rel, w_rel, h_rel = det.bounding_box

                # Compute corner coordinates (still relative)
                x1 = x_rel
                y1 = y_rel
                x2 = x_rel + w_rel
                y2 = y_rel + h_rel

                # Format: class_id x1 y1 x2 y1 x2 y2 x1 y2
                label_line = f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
                label_lines.append(label_line)

            # Save to file with same name as image
            img_name = os.path.splitext(os.path.basename(sample.filepath))[0]
            label_path = os.path.join(annotations_dir, f"{img_name}.txt")

            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

        print(f"Saved bounding box files to '{annotations_dir}'.")



# Define mapping from class name to numeric ID
class_map = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Truck": 3,
    "Van": 4
}

class_map_rev = {v: k for k, v in class_map.items()}