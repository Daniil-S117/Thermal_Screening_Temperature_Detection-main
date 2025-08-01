import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to read camera feed")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(round(cap.get(5)))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while (True):
    ret, frame = cap.read()

    if ret == True:
        cv2.putText(frame, 'hello',
                    org=(10, 300),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=5,
                    color=(255, 0, 0),
                    thickness=4)
        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()