# object-tracking

I used YOLOv3 pretrained network to track objects in a video. In code file you have to put your video file path.

Download YOLOv3 weights from YOLO website:  https://pjreddie.com/darknet/yolo/

or using following command:- 

```bash
wget https://pjreddie.com/media/files/yolov3.weights
```

After downloading put the weight file in model_data folder. Now run following command to convert these weights for Keras model.

```bash
python convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo_weights.h5
```

Now we have everything to run the model. 

Requirements:-

1. Keras
2. Tensorflow
3. PIL
4. Matplotlib
5. Numpy
6. Opencv



Use good GPU system to avoid video lagging. Nvidia GTX 1050ti would be good.

After doing everything run it by:

`python` `yolo_object_detection.py`

