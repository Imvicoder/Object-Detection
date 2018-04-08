import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option={
     'model':'cfg/yolo.cfg',
     'load':'bin/yolov2.weights',
     'threshold':0.15 
}
tfnet=TFNet(option)
capture=cv2.VideoCapture('test.mp4')
colors=[tuple(255*np.random.rand(3)) for i in range(5)]

#for color in colors:
 #   print (color)
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('videofile_1080_20fps.avi', codec, 60.0, size)
i = 0
frame_rate_divider = 3
while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break