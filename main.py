import cv2
from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture("test.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (
cv2.CAP_PROP_FRAME_WIDTH, 
cv2.CAP_PROP_FRAME_HEIGHT, 
cv2.CAP_PROP_FPS))

# w, h = 640, 480
# fps = int(cap.get(cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(
"speed_estimate.mp4",
cv2.VideoWriter_fourcc(*"mp4v"),
fps, (w, h))

line_pts = [(0, h // 2), (w, h // 2)]

speed_obj = solutions.SpeedEstimator(reg_pts=line_pts,
                                      names=names,
                                      view_img=True,)

while cap.isOpened():
  success, im0 = cap.read()
  if not success:        
    break
  tracks = model.track(im0, persist=True, show=False)
  im0 = speed_obj.estimate_speed(im0, tracks)
  video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()