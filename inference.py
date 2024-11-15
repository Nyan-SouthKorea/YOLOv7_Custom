import torch
import cv2
import numpy as np
import time

class Custom_YOLOv7:
    '''
    데모 코드
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # 너비 설정
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # 높이 설정
class_list = ['person', 'dog', 'cat']

model = Custom_YOLOv7('best.pt', class_list)
while True:
    ret, img = cap.read()
    if ret == False:
        print('웹캠 문제')
        break
    dic_list = model.predict(img)
    cv2.imshow('show', model.draw(img, dic_list))
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
    '''
    def __init__(self, model_path, class_list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path).to(self.device).eval()
        self.class_list = class_list
        self.fps = 0

    def predict(self, bgr_img, conf_thres=0.25, iou_thres=0.45):
        start = time.time()
        h, w = 640, 640
        image_resized = cv2.resize(bgr_img, (640, 640))
        h, w, _ = image_resized.shape
        image_resized = image_resized[:, :, ::-1].copy()
        image_resized = np.transpose(image_resized, (2, 0, 1))
        image_resized = np.expand_dims(image_resized, axis=0)
        image_resized = torch.from_numpy(image_resized).float().to(self.device) / 255.0
        
        # 추론
        with torch.no_grad():
            detections = self.model(image_resized)
        results = detections[0].cpu().numpy()
        
        # 3차원 배열인 경우 모든 스케일과 출력값을 2차원으로 병합
        if results.ndim == 3:
            results = results.reshape(-1, results.shape[-1])

        # Confidence 기준으로 필터링
        filtered_results = results[results[:, 4] >= conf_thres]
        
        # dic_list 생성
        dic_list = []
        for detection in filtered_results:
            conf = round(float(detection[4]), 2)
            cx, cy, width, height = detection[:4]
            x1 = int(cx - width / 2)
            y1 = int(cy - height / 2)
            x2 = int(cx + width / 2)
            y2 = int(cy + height / 2)

            # 정규화
            r_no = 5
            x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
            x1, y1, x2, y2 = round(x1, r_no), round(y1, r_no), round(x2, r_no), round(y2, r_no)
            probabilities = detection[5:]
            class_id = int(np.argmax(probabilities))
            class_name = self.class_list[class_id]
            dic_list.append({
                'class_no': class_id,
                'class_name': class_name,
                'conf': conf,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'del':False
            })
        self.fps = int(1 / (time.time()-start+0.000001))
        return self.nms_dic_list(dic_list, iou_thres)
    
    def nms_dic_list(self, dic_list, iou_thres=0.45):
        nms_list = []
        for i1, dic1 in enumerate(dic_list):
            x1, y1, x2, y2, conf = dic1['x1'], dic1['y1'], dic1['x2'], dic1['y2'], dic1['conf']
            bbox1 = [x1, y1, x2, y2, conf]
            for i2, dic2 in enumerate(dic_list):
                x1, y1, x2, y2, conf = dic2['x1'], dic2['y1'], dic2['x2'], dic2['y2'], dic2['conf']
                bbox2 = [x1, y1, x2, y2, conf]
                if self.iou(bbox1, bbox2) >= iou_thres:
                    if bbox1[4] > bbox2[4]:
                        dic_list[i2]['del'] = True
        nms_list = []
        for dic in dic_list:
            if dic['del'] == False:
                del dic['del']
                nms_list.append(dic)
        return nms_list

    def iou(self, bbox_a, bbox_b):
        '''
        입력된 bbox 2개 대하여 iou 출력
        '''
        a_x1, a_y1, a_x2, a_y2, a_conf = bbox_a
        b_x1, b_y1, b_x2, b_y2, b_conf = bbox_b
        # 작은 박스 구하기
        small_x1 = max(a_x1, b_x1)
        small_y1 = max(a_y1, b_y1)
        small_x2 = min(a_x2, b_x2)
        small_y2 = min(a_y2, b_y2)
        width_small = small_x2 - small_x1
        height_small = small_y2 - small_y1
        if width_small <= 0 or height_small <= 0: # 박스가 겹치지 않는다
            return 0.0
        area_small = width_small * height_small

        # box a 면적 구하기
        width_a = a_x2 - a_x1
        height_a = a_y2 - a_y1
        area_a = width_a * height_a
        
        # box b 면적 구하기
        width_b = b_x2 - b_x1
        height_b = b_y2 - b_y1
        area_b = width_b * height_b

        # IOU 구하기
        iou_down = area_a + area_b - area_small
        iou = area_small / iou_down
        return iou

    def draw(self, img, dic_list):
        h, w, _ = img.shape
        color, thick, txt_size = [0,0,255], 2, 1
        for dic in dic_list:
            x1, y1, x2, y2 = dic['x1'], dic['y1'], dic['x2'], dic['y2']
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)
            txt = f'{dic["class_name"]}:{dic["conf"]}'
            cv2.putText(img, txt, (x1,y2-3), cv2.FONT_HERSHEY_SIMPLEX, txt_size, color, thick)
        cv2.putText(img, f'FPS:{self.fps}', (3,30), cv2.FONT_HERSHEY_SIMPLEX, txt_size, color, thick)
        return img

def smart_resize(img, max_size=1280):
    '''
    최대 변의 길이를 맞추면서 비율을 유지하여 이미지 리사이즈
    img: cv2 이미지
    max_size: 최대 크기
    return: resize된 cv2 이미지 반환
    '''
    h, w, _ = img.shape
    if w > h:
        img = cv2.resize(img, (max_size, int(h/w*max_size)))
    else:
        img = cv2.resize(img, (int(w/h*max_size), max_size))
    return img


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # 너비 설정
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # 높이 설정
class_list = ['person', 'dog', 'cat']

model = Custom_YOLOv7('best.pt', class_list)
while True:
    ret, img = cap.read()
    if ret == False:
        print('웹캠 문제')
        break
    dic_list = model.predict(img)
    cv2.imshow('show', model.draw(img, dic_list))
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()

