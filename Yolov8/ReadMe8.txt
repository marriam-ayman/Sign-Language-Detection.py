//to run Yolov8 
make a new folder in visual studio code and make file rename it to main.py and paste this code in it 

from ultralytics import YOLO
import cv2

model=YOLO("best ''''''(4).pt")

results= model.predict(source="0",show=True)

print(results)


and open the file of best (4).pt (this is the trained file) 
then the webcam will open and you can test it 