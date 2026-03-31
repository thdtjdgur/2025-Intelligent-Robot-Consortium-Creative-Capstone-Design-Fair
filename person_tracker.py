import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

from std_msgs.msg import String ##########bluetooth
import rospy#################bluetooth

# === YOLO 파일 경로 설정 ===
yolo_path = os.path.expanduser("~/yolot")  # 니가 파일 넣어둔 폴더

weights_path = os.path.join(yolo_path, "yolov4-tiny.weights")
cfg_path     = os.path.join(yolo_path, "yolov4-tiny.cfg")
names_path   = os.path.join(yolo_path, "coco.names")

# === YOLO 모델 로딩 + GPU 사용 설정 ===
net = cv2.dnn.readNet(weights_path, cfg_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

speed_up_flag = False
speed_down_flag = False
move_right_flag = False
move_left_flag = False

prev_speed_up = False
prev_speed_down = False
prev_right = False
prev_left = False



# === 클래스 이름 로딩 ===
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# === RealSense 카메라 설정 ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


#######################################################################bluetooth
rospy.init_node("yolo_bt_client", anonymous=False)
bt_pub = rospy.Publisher("/bt_cmd", String, queue_size=10)

last_dist_cmd = None
last_lat_cmd = None
last_tx_ts = 0.0
MIN_TX_INTERVAL= 0.10

def send_cmd(cmd: str):
    global last_tx_ts
    now = time.time()
    if now - last_tx_ts >=MIN_TX_INTERVAL:
        bt_pub.publish(cmd)
        last_tx_ts = now





# === 지속 조건(딜레이) 설정 ===
ACTIVATION_DELAY = 2.0  # ← 2초 이상 연속으로 조건 만족 시 플래그 ON    #################   flag time eeeeeeeee

# 조건 누적 타이머들
acc_far = 0.0     # dist >= 1.3  지속 시간
acc_near = 0.0    # dist < 0.9   지속 시간
acc_left = 0.0    # cx < left_bound  지속 시간
acc_right = 0.0   # cx > right_bound 지속 시간
last_ts = time.time()

# === 최적화 설정 ===
frame_count = 0
yolo_results = []
skip_interval = 3  # 프레임 스킵 간격 (3프레임에 한 번)

try:
    while True:
        now = time.time()
        dt = now - last_ts          # 프레임 간 시간
        last_ts = now

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        height, width = color_image.shape[:2]

        frame_count += 1

        # [ADD] 좌우 기준선/안전 구간 계산(프레임마다)
        center_x = width // 2
        zone = int(width * 0.15)          # 안전 구간 ±15% (원하면 조정)    ########################3   from center, how many distance range set
        left_bound  = center_x - zone
        right_bound = center_x + zone

        # 이번 프레임 플래그 기본값 초기화(최종 출력용)
        speed_up_flag = False
        speed_down_flag = False
        move_right_flag = False
        move_left_flag  = False

        # === YOLO 프레임 스킵 처리 ===
        if frame_count % skip_interval == 0:
            start = time.time()

            # 입력 해상도 축소 (320x320)
            blob = cv2.dnn.blobFromImage(color_image, 1/255.0, (320, 320), swapRB=True, crop=False)
            net.setInput(blob)
            yolo_results = net.forward(output_layers)

            #print(f"YOLO 추론시간: {time.time() - start:.3f}s")

        # === 탐지 결과 처리 ===
        boxes, confidences, class_ids = [], [], []

        for out in yolo_results:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "person":
                    center_x_det = int(detection[0] * width)
                    center_y_det = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x_det - w // 2)
                    y = int(center_y_det - h // 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # [ADD] 가장 가까운 사람 찾기용 변수
        nearest = None   # (distance, cx, cy, x, y, w, h, label, conf)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                conf = confidences[i]

                # 중심점/거리
                cx, cy = x + w // 2, y + h // 2
                distance = depth_frame.get_distance(cx, cy)

                # 가장 가까운 사람 갱신(유효 거리만)
                if distance > 0:
                    if (nearest is None) or (distance < nearest[0]):
                        nearest = (distance, cx, cy, x, y, w, h, label, conf)

                # 시각화
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label_text = f"{label} {conf:.2f}"
                label_y = max(20, y - 10)
                cv2.putText(color_image, label_text, (x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                dist_text = f"{distance:.2f} m"
                cv2.putText(color_image, dist_text, (x + tw + 8, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # === 지속 조건(2초) 적용한 플래그 판정 ===
        if nearest is not None:
            dist, cx, cy, x, y, w, h, label, conf = nearest

            # 1) 원시 조건 계산
            is_far  = (dist >= 1.3)      ##############  distance far
            is_near = (dist < 0.9)       ########distance close
            is_left_side  = (cx < left_bound)   # 화면 왼쪽 바깥
            is_right_side = (cx > right_bound)  # 화면 오른쪽 바깥

            # 2) 누적(연속) 시간 갱신
            acc_far  = acc_far  + dt if is_far  else 0.0
            acc_near = acc_near + dt if is_near else 0.0
            acc_left = acc_left + dt if is_left_side  else 0.0
            acc_right= acc_right+ dt if is_right_side else 0.0

            # 3) 누적 시간이 임계치를 넘으면 플래그 ON
            speed_up_flag   = (acc_far  >= ACTIVATION_DELAY)
            speed_down_flag = (acc_near >= ACTIVATION_DELAY)
            move_right_flag = (acc_left >= ACTIVATION_DELAY)   # 왼쪽으로 치우침 지속 → 오른쪽 이동 안내
            move_left_flag  = (acc_right>= ACTIVATION_DELAY)   # 오른쪽으로 치우침 지속 → 왼쪽 이동 안내

            # 디버그 출력(선택)
            if speed_up_flag and not prev_speed_up:
                send_cmd("up")
                print("speed up (held >= 2s)")
            elif speed_down_flag and not prev_speed_down:
                send_cmd("down")
                print("slow down (held >= 2s)")
            #else:
                #print("keep running")

            if move_left_flag and not prev_left:
                send_cmd("right")
                print("go right (held >= 2s)")
            elif move_right_flag and not prev_right:
                send_cmd("left")
                print("go left (held >= 2s)")
            #else:
                #print("center")
               
               
            prev_speed_up = speed_up_flag
            prev_speed_down = speed_down_flag
            prev_left = move_left_flag
            prev_right = move_right_flag
               
        else:
            prev_speed_up = False
            prev_speed_down = False
            prev_left = False
            prev_right = False
       
       
            # 타깃 없음 -> 누적 타이머/플래그 리셋
            acc_far = acc_near = acc_left = acc_right = 0.0
            speed_up_flag = speed_down_flag = False
            move_left_flag = move_right_flag = False

        cv2.imshow("YOLOv4-tiny + RealSense (최적화)", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()