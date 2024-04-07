import cv2
import time
import serial
import numpy as np
import supervision as sv

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
STATE = '0'
DESIRED_CLASS_ID

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
        DESIRED_CLASS_ID = 1
        return cap0.read()
    else:
        DESIRED_CLASS_ID = 3
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

def silo0_available(count):
    if count > 0:
        return True
    return False

def silo2_available(count):
    if count > 0: 
        return True
    return False

def silo1_t_available(count):
    if count > 0:
        return True
    return False

def count_objects_with_name(frame, name):
    # Load pre-trained YOLOv8 model

    result = yolov8_detector(frame, agnostic_nms=True, conf=0.5)[0]


    detections = sv.Detections.from_ultralytics(result)

    count = 0
    for obj in detections:
        print(obj[5]['class_name'])
        if (obj[5]['class_name'] == name):
          count += 1
    
    return count
#cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while True:

    # Read frame from the video
    ret, frame = get_frame()

    

    if not ret:
        break
    
    boxes, scores, class_ids = detect(frame)

    if STATE == '0':
        DESIRED_CLASS_ID = 1
    elif STATE == '1':
        count_silo0 = count_objects_with_name(frame, 'silo0')
        count_silo1_t = count_objects_with_name(frame, 'silo1_b')
        count_silo2 = count_objects_with_name(frame, 'silo2')
        if silo2_available(count_silo2):
            DESIRED_CLASS_ID = 6
        elif silo1_t_available(count_silo1_t):
            DESIRED_CLASS_ID = 4
        elif silo0_available(count_silo0):
            DESIRED_CLASS_ID = 3
    
    filtered_boxes = []
    filtered_scores = []
    filtered_class_ids = []
    
    for i in range(len(boxes)):
        if class_ids[i] == DESIRED_CLASS_ID:  
            filtered_boxes.append(boxes[i])
            filtered_scores.append(scores[i])
            filtered_class_ids.append(class_ids[i])

        
    
    #write_data(filtered_boxes)
    if len(filtered_boxes) > 0:
        id_get = np.argmax((np.array(filtered_boxes)[:,2]-np.array(filtered_boxes)[:,0]) * (np.array(filtered_boxes)[:,3]-np.array(filtered_boxes)[:,1]))
        x, y = (filtered_boxes[id_get][2] + filtered_boxes[id_get][0])//2, (filtered_boxes[id_get][3] + filtered_boxes[id_get][1])//2

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
