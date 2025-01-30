import numpy as np
import supervision as sv
from ultralytics import YOLO
import argparse
import cv2

parser = argparse.ArgumentParser(
    prog='yolov8',
    description='This program helps detect and count people in the polygon region.',
    epilog='Text at the bottom of help')

parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)

args = parser.parse_args()

class CountObject():
    def __init__(self, input_video_path, output_video_path) -> None:
        self.model = YOLO('yolov8s.pt')
        self.colors = sv.ColorPalette.default()

        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.polygons = []  # Will hold polygons created by the user

        self.video_info = sv.VideoInfo.from_video_path(input_video_path)

        self.zones = []  # Initialize zones as empty
        self.zone_annotators = []
        self.box_annotators = []

        self.frame = None  # Initialize frame as None

    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygons[-1].append((x, y))
            frame_copy = self.frame.copy()
            for i in range(len(self.polygons[-1]) - 1):
                cv2.line(frame_copy, self.polygons[-1][i], self.polygons[-1][i+1], (0, 255, 0), 2)
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
            if len(self.polygons[-1]) > 1:
                cv2.line(frame_copy, self.polygons[-1][-2], (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw Polygons", frame_copy)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.polygons[-1]) >= 3:  # Require at least 3 points to close the polygon
                # Close the polygon by connecting to the first point
                self.polygons[-1].append(self.polygons[-1][0])
                self.create_zone()
                # Prepare for new polygon
                self.polygons.append([])
            else:
                print("A polygon requires at least 3 points. Please add more points.")

    def create_zone(self):
        # Use the last polygon which is now closed
        polygon = np.array(self.polygons[-1], np.int32)
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=self.video_info.resolution_wh)
        self.zones.append(zone)

        self.zone_annotators.append(
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=self.colors.by_idx(len(self.zones)-1),
                thickness=6,
                text_thickness=8,
                text_scale=4
            )
        )

        self.box_annotators.append(
            sv.BoxAnnotator(
                color=self.colors.by_idx(len(self.zones)-1),
                thickness=4,
                text_thickness=4,
                text_scale=2
            )
        )

    def process_frame(self, frame: np.ndarray, i) -> np.ndarray:
        results = self.model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_yolov8(results)
        detections = detections[(detections.class_id == 2) | (detections.class_id == 5)]
        detections = detections[detections.confidence > 0.5]

        for zone, zone_annotator, box_annotator in zip(self.zones, self.zone_annotators, self.box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
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
                    if len(polygon) >= 3:  # Only close if enough points
                        cv2.line(frame_copy, polygon[-1], polygon[0], (0, 255, 0), 2)
            cv2.imshow("Draw Polygons", frame_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('n'):
                self.polygons.append([])

        cap.release()
        cv2.destroyAllWindows()

        # Process video with the created zones
        sv.process_video(source_path=self.input_video_path, target_path=self.output_video_path, callback=self.process_frame)

if __name__ == "__main__":
    obj = CountObject(args.input, args.output)
    obj.process_video()