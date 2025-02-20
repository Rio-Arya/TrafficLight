import numpy as np
import supervision as sv
from ultralytics import YOLO
import argparse
import cv2

def get_vehicle_weight(class_id):
    weight_map = {
        3: 1,  # Motor
        2: 2,  # Mobil
        5: 4,  # Bis
        7: 5   # Truk
    }
    return weight_map.get(class_id, 0)  # Default to 0 if class not found

parser = argparse.ArgumentParser(
    prog='yolov8',
    description='Detect, count, and calculate vehicle weight in a defined region.',
    epilog='Press Q to exit the polygon selection mode.')

parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

class CountObject():
    def __init__(self, input_video_path, output_video_path) -> None:
        self.model = YOLO('yolov8s.pt')
        self.colors = sv.ColorPalette.default()
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.polygons = []
        self.video_info = sv.VideoInfo.from_video_path(input_video_path)
        self.zones = []
        self.zone_annotators = []
        self.box_annotators = []
        self.frame = None

    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygons[-1].append((x, y))
            frame_copy = self.frame.copy()
            for i in range(len(self.polygons[-1]) - 1):
                cv2.line(frame_copy, self.polygons[-1][i], self.polygons[-1][i+1], (0, 255, 0), 1)
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
            if len(self.polygons[-1]) > 1:
                cv2.line(frame_copy, self.polygons[-1][-2], (x, y), (0, 255, 0), 1)
            cv2.imshow("Draw Polygons", frame_copy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.polygons[-1]) >= 3:
                self.polygons[-1].append(self.polygons[-1][0])
                self.create_zone()
                self.polygons.append([])
            else:
                print("A polygon requires at least 3 points. Please add more points.")

    def create_zone(self):
        polygon = np.array(self.polygons[-1], np.int32)
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=self.video_info.resolution_wh)
        self.zones.append(zone)
        self.zone_annotators.append(
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=self.colors.by_idx(len(self.zones)-1),
                thickness=3,
                text_thickness=0,  # Nonaktifkan teks object counting
                text_scale=0       # Nonaktifkan teks object counting
            )
        )
        self.box_annotators.append(
            sv.BoxAnnotator(
                color=self.colors.by_idx(len(self.zones)-1),
                thickness=4,
            )
        )

    def process_frame(self, frame: np.ndarray, i) -> np.ndarray:
        results = self.model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_yolov8(results)

        if len(detections.xyxy) == 0:  # Ensure there are detections
            return frame  # Return the frame unchanged if no detections

        detections = detections[detections.confidence > 0.5]

        for zone, zone_annotator, box_annotator in zip(self.zones, self.zone_annotators, self.box_annotators):
            mask = zone.trigger(detections=detections)

            if mask is None or len(mask) == 0:
                continue  # Skip if no detections in this zone

            detections_filtered = detections[mask]

            if len(detections_filtered.xyxy) == 0:
                continue  # Skip if no objects detected in this zone

            # Using a dictionary to count the number of each class_id in one polygon
            zone_weights = {}

            for det in detections_filtered:
                class_id = det[2]  # Access class_id from the detections tuple
                if class_id is None:
                    continue  # Skip if no class_id

                class_id = int(class_id)  
                zone_weights[class_id] = zone_weights.get(class_id, 0) + 1  # Count vehicles per class_id

            # Summing the total weight of vehicles in one polygon
            zone_total_weight = sum(get_vehicle_weight(class_id) * count for class_id, count in zone_weights.items())

            # Display the total weight in the zone
            centroid = np.mean(zone.polygon, axis=0).astype(int)
            cv2.putText(frame, f"Weight: {zone_total_weight}", (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw bounding boxes without labels
            for bbox in detections_filtered.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                # Get BGR color from sv.Color
                color = self.colors.by_idx(len(self.zones)-1)
                bgr_color = color.as_bgr()  # Convert to BGR format
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)

            frame = zone_annotator.annotate(scene=frame)

        return frame

    def process_video(self):
        cap = cv2.VideoCapture(self.input_video_path)
        ret, self.frame = cap.read()
        self.polygons.append([])
        cv2.namedWindow("Draw Polygons")
        cv2.setMouseCallback("Draw Polygons", self.draw_polygon)
        
        while True:
            frame_copy = self.frame.copy()
            for polygon in self.polygons:
                if len(polygon) > 1:
                    for i in range(len(polygon) - 1):
                        cv2.line(frame_copy, polygon[i], polygon[i+1], (0, 255, 0), 2)
                    if len(polygon) >= 3:
                        cv2.line(frame_copy, polygon[-1], polygon[0], (0, 255, 0), 2)
            cv2.imshow("Draw Polygons", frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n'):
                self.polygons.append([])
        
        cap.release()
        cv2.destroyAllWindows()
        sv.process_video(source_path=self.input_video_path, target_path=self.output_video_path, callback=self.process_frame)

if __name__ == "__main__":
    obj = CountObject(args.input, args.output)
    obj.process_video()