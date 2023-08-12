import cv2
import lanmarks

cap = cv2.VideoCapture('./videos/t.mp4')
lan = lanmarks.Landmarks(960,540)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame = cv2.resize(frame, (960,540))

    points = lan.get(frame, 'lip')
    print(points)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
