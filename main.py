#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0                        NAM MÔ A DI ĐÀ PHẬT
#                    ___/`---'\___              
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#            Phật phù hộ, không bao giờ Bug
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import cv2
import time
import serial
import numpy as np

from yolov8 import YOLOv8, draw_detections
# from tracking import BYTETracker

# Initialize the webcam
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Initialize YOLOv7 object detector
model_path = "weights/best243_360.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
ser = serial.Serial('/dev/ttyUSB0', 115200)
# tracker = BYTETracker()
STATE_ball = '0'
STATE = '0'

def get_state():
    if ser.inWaiting() == 0:
            #print(0)
            return STATE

    # Đoạn này trở đi là nhan lenh tu serial
    data1 = ser.readline(1)
    print(data1)
    try:
        data1 = str(data1, 'utf-8')
    except UnicodeDecodeError:
        # Xử lý khi gặp lỗi UnicodeDecodeError
        data1 = str(data1, 'latin-1')  # hoặc mã hóa khác
    print(data1.strip('\r\n'))
    return data1.strip('\r\n')

def get_frame():
    global STATE
    current_state = get_state()
    STATE = current_state if current_state != STATE and current_state != '' else STATE
    if STATE == '0':
        return cap0.read()
    else:
        return cap1.read()

def detect(frame):
    start_time = time.time()
    boxes, scores, class_ids = yolov8_detector(frame)
    print(f'{(time.time() - start_time)*1000:.2f} ms')
    
    # boxes, scores, class_ids, ids = tracker.predict(frame, boxes, scores, class_ids)

    return boxes, scores, class_ids
    
def write_data(filtered_boxes):
    if len(filtered_boxes) > 0:
        id_get = np.argmax((np.array(filtered_boxes)[:,2]-np.array(filtered_boxes)[:,0]) * (np.array(filtered_boxes)[:,3]-np.array(filtered_boxes)[:,1]))
        x, y = (filtered_boxes[id_get][2] + filtered_boxes[id_get][0])//2, (filtered_boxes[id_get][3] + filtered_boxes[id_get][1])//2

        data = str(x)+','+str(y)+'\r'
        ser.write(data.encode())
    else:
        x, y = 0, 0
        data = str(x)+','+str(y)+'\r'
        ser.write(data.encode())

#đếm số lượng của từng loại silo
'''
Quy ước: 
['ball_r', 'ball_b', 'silo0', 'silo1_r', 'silo1_b', 'silo2']
[  '!',       '!',      '!',      '!',      '!',      '!']
[  '0',       '1',      '2',      '3',      '4',      '5']
'''
def count_object_each_class_id(class_ids):
  frequency = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
  for id in class_ids:
    if 0 <= id <= 5:
      frequency[id] += 1
  return frequency

#Hàm lọc box theo id tương ứng: 
#Hàm lọc box theo id tương ứng: 
def filter_boxes(boxes, class_ids, id):
  boxes_class_id = []  # Danh sách lưu trữ boxes có class_id = id
  class_ids_class_id = []

  for box, class_id in zip(boxes, class_ids):
    if class_id == id:
      boxes_class_id.append(box)
      class_ids_class_id.append(class_id)

  return boxes_class_id, class_ids_class_id

#Get type ball
'''
Quy ước: 
STATE_ball = '0' với bóng đỏ (r)
STATE_ball = '1' vơi bóng xanh (b)
'''
def get_type_ball():
    STATE_ball = get_state()
    ball_id = 0
    if STATE_ball == '0':
        ball_id = 0
    else:
        ball_id = 1
    
    return ball_id

def sort_boxes_by_area(boxes, center_x, center_y):

    areas = (np.array(boxes)[:, 2] - np.array(boxes)[:, 0]) * \
            (np.array(boxes)[:, 3] - np.array(boxes)[:, 1])

    sorted_boxes = sorted(zip(boxes, areas, range(len(boxes))), key=lambda x: x[1], reverse=True)
    # n% = 20%
    if (sorted_boxes[0][1]/sorted_boxes[1][1] > 1.2):
        return sorted_boxes[0]
    else:
        id_get = np.argmin(np.sqrt(((np.array(boxes)[:,2]+np.array(boxes)[:,0])//2-center_x)**2 + ((np.array(boxes)[:,3]+np.array(boxes)[:,1])//2-center_y)**2))
        return sorted_boxes[id_get]

    #return sorted_boxes


type_ball = get_type_ball()

#cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while True:

    # Read frame from the video
    ret, frame = get_frame()

    image_height, image_width, _ = frame.shape
    center_x = image_width // 2
    center_y = image_height // 2

    if not ret:
        break
    
    boxes, scores, class_ids = detect(frame)

    #Phân loại từng loại tương ứng với từng trường hợp trong 
    if STATE == '0':
        boxes, class_ids = filter_boxes(boxes, class_ids, type_ball)
    elif STATE == '1':
        freq = count_object_each_class_id(class_ids)
        if (freq[5] > 0):
            boxes, class_ids = filter_boxes(boxes, class_ids, 5)
        elif (freq[type_ball] > 0):
            boxes, class_ids = filter_boxes(boxes, class_ids, type_ball)
        else:
            boxes, class_ids = filter_boxes(boxes, class_ids, 2)

    
    #write_data(filtered_boxes)
    if len(boxes) > 1:

        boxes = sort_boxes_by_area(boxes, center_x, center_y)
        id_get = 0
        x, y = (boxes[id_get][2] + boxes[id_get][0])//2, (boxes[id_get][3] + boxes[id_get][1])//2
        data = str(x)+','+str(y)+'\r'
        ser.write(data.encode())
    elif len(boxes) == 1:
        id_get = 0
        x, y = (boxes[id_get][2] + boxes[id_get][0])//2, (boxes[id_get][3] + boxes[id_get][1])//2
        data = str(x)+','+str(y)+'\r'
        ser.write(data.encode())
    else:
        x, y = 0, 0
        data = str(x)+','+str(y)+'\r'
        ser.write(data.encode())
	
    #print(data)
    combined_img = draw_detections(frame, boxes, scores, class_ids)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release() 
cap1.release() 
cv2.destroyAllWindows()
