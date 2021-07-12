import cv2
from backend.ssd_detection import Detector as Detector_SSD
from backend.yolo_detection import Detector as Detector_Yolo
from backend.motion import Detector as Detector_Motion
from backend.cascade import Detector as Detector_Cascade
from utils.torch_utils import select_device, load_classifier, time_synchronized

def test_ssd():
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector_SSD()
    t1 = time_synchronized()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    t2 = time_synchronized()
    print('detect time: ', t2-t1)
    image = detector.draw_boxes(image, df)
    print(df)
    assert df.shape[0] == 2
    assert any(df['class_name'].str.contains('person'))
    assert any(df['class_name'].str.contains('dog'))
    cv2.imwrite("./imgs/outputcv.jpg", image)


def test_yolo():
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector_Yolo()
    t1 = time_synchronized()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    t2 = time_synchronized()
    print('detect time: ', t2-t1)
    image = detector.draw_boxes(image, df)
    print(df)
    assert df.shape[0] == 1
    assert any(df['class_name'].str.contains('dog'))
    cv2.imwrite("./imgs/outputcv.jpg", image)


def test_motion():
    image = cv2.imread("./imgs/image.jpeg")
    print(image.shape)

    detector = Detector_Motion()

    image2 = cv2.imread("./imgs/image_box.jpg")
    print(image2.shape)
    assert image.shape == image2.shape
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.GaussianBlur(image2, (21, 21), 0)
    detector.avg = image2.astype(float)

    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    image = detector.draw_boxes(image, df)
    print(df)
    assert df.shape[0] == 1

    cv2.imwrite("./imgs/outputcv.jpg", image)


def test_cascade():
    image = cv2.imread("./imgs/image.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = Detector_Cascade()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    image = detector.draw_boxes(image, df)

    print(df)
    #cv2.imwrite("./imgs/outputcv.jpg", image)
test_ssd()
test_yolo()
