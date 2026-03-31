#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, random, math, copy, time
from std_msgs.msg import Int32

Width = 640 #640
Height = 480 #480
warp_Offset = 120
lane_bin_th_l, lane_bin_th_r = 120, 120    
prev_steer_angle = 0

prev_c0, prev_c1= 0,0

gst_pipeline = (                           # 여기 바
    "nvarguscamerasrc sensor_mode=4 ! "
    "video/x-raw(memory:NVMM), width=640, height=480, framerate=21/1, format=NV12 ! "
    "queue max-size-buffers=1 leaky=downstream ! "  
    "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=true"
)

warp_img_w = Width // 2  #320          
warp_img_h = Height // 2 #240

warpx_margin = 20   #40      
warpy_margin = 3   #5

# 슬라이딩 윈도우 개수값
nwindows = 9
# 슬라이딩 윈도우 넓이
margin = 30 #12    ----> 이게 넓어야 라인 전체를 파악해서 정확한 조향각 설정 가능

minpix = 5

lane_bin_th = 125

warp_src = np.array([                # 너무 범위가 크면 펴주는 비율이 별로 안되서 버드아이뷰가 적용이 안됨 좁고 길게 좌표 설정
    [150,330], #[90,320],   #[30,355],   #[60, 532],    
    [46,379], #[20,420],   #[0,385],      #[0, 577],
    [480,330], #[550,320],   #[Width-45, 355],   #[Width-90, 532],
    [596,379] #[620,420]   #[Width, 385]   #[Width, 577]
], dtype=np.float32)

warp_dist = np.array([  
    [0,0],
    [0,warp_img_h],
    [warp_img_w,0],
    [warp_img_w, warp_img_h]
], dtype=np.float32)


M = cv2.getPerspectiveTransform(warp_src, warp_dist)
Minv = cv2.getPerspectiveTransform(warp_dist, warp_src)

calibrated = True

if calibrated:
   
    mtx = np.array([         
       
       
       [15.65809020e+03, 0.00000000e+00, 6.18013039e+02],
        [0.00000000e+00, 1.71372547e+03, 2.96835808e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],            
])

    dist = np.array([-1.54438124e+00, 3.29679370e+00, 1.88881286e-02,-3.75227931e-03, -4.84598887e+00])

 
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))

    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, cal_mtx, (Width,Height),cv2.CV_16SC2)

   

def calibrate_image(frame):
   
    tf_image = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = cal_roi  
    tf_image = tf_image[y:y+h, x:x+w]

    return cv2.resize(tf_image, (Width, Height))


def warp_image(img):  
    warp_img = cv2.warpPerspective(img, M, (int(warp_img_w), int(warp_img_h)), flags=cv2.INTER_LINEAR)

    return warp_img


def homomorphic_filter(img, gamma_low=0.05, gamma_high=1.3, c=1.8, d0=32):             #dark: gamma high=1.3 good            # gamma_low=0.3, gamma_high=2.8, c=1.8, d0=20   #(img, gamma_low=0.5, gamma_high=2.0, c=1.0, d0=30):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()


    pad=10
    img_gray = img_gray.astype(np.float32) + 1.0
    log_img = np.log(img_gray)

    fft_img = np.fft.fft2(log_img)
    fft_shift = np.fft.fftshift(fft_img)

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    # 벡터화된 고역통과 필터 계산
    U, V = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    Duv = np.sqrt((U - crow) ** 2 + (V - ccol) ** 2)
    H = (gamma_high - gamma_low) * (1 - np.exp(-c * (Duv ** 2) / (d0 ** 2))) + gamma_low

    filtered_fft = fft_shift * H
    ifft_shift = np.fft.ifftshift(filtered_fft)
    img_back = np.fft.ifft2(ifft_shift)
    img_back = np.real(img_back)

    img_exp = np.exp(img_back) - 1.0
   
    img_exp_cropped = img_exp[pad:-pad, pad:-pad]
   
    img_out = cv2.normalize(img_exp_cropped, None, 0, 255, cv2.NORM_MINMAX)
    img_out = img_out.astype(np.uint8)

    cv2.imshow("homomorphic_filter",img_out)
    return img_out




def warp_process_image(img):
    global prev_c0,  prev_c1
    # 0. Homomorphic Filtering
    img_homo = homomorphic_filter(img)
    img_homo_bgr = cv2.cvtColor(img_homo, cv2.COLOR_GRAY2BGR)
    cv2.imshow("homo",img_homo_bgr)

    # 1. Gaussian Blur
    blur = cv2.GaussianBlur(img_homo_bgr, (5, 5), 0)

    # 2. Extract L channel from HLS
    _, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))

    # 3. Adaptive thresholding for left/right halves
    lane_bin_th_l = L[:, :(warp_img_w // 2)].mean() + 20
    lane_bin_th_r = L[:, (warp_img_w // 2):].mean() + 20

    _, lane_l = cv2.threshold(L, lane_bin_th_l, 255, cv2.THRESH_BINARY)
    _, lane_r = cv2.threshold(L, lane_bin_th_r, 255, cv2.THRESH_BINARY)

    lane_L = np.concatenate((lane_l[:, :warp_img_w // 2], lane_r[:, warp_img_w // 2:]), axis=1)
    lane = lane_L.copy()
    cv2.imshow("test",lane)
   
    # 4. Morphology
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))    #cv2.MORPH_RECT,55s
    lane = cv2.morphologyEx(lane, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, kernel_close)
    cv2.imshow("mor",lane)

    # 5. Contour filtering for noise removal
    contours, _ = cv2.findContours(lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        mask = np.zeros_like(lane)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        roi = cv2.bitwise_and(lane, mask)

        vertical_profile = np.sum(roi, axis=1) // 255
        consecutive_white_rows = np.count_nonzero(vertical_profile > 10)

        if consecutive_white_rows > 30:
            continue

        if area < 200:
            cv2.drawContours(lane, [cnt], -1, 0, -1)
            #cv2.imshow("cop",lane)
    cv2.imshow("thr",lane)
    # 6. Sliding window with histogram + contour hybrid initialization
    #histogram = np.sum(lane[lane.shape[0] // 2:, :], axis=0)

    # === New contour-based initial window centers ===
    contours, _ = cv2.findContours(lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_centers = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 500:
            cx = x + w // 2
            contour_centers.append(cx)


    contour_centers = sorted(contour_centers)
    leftx_current, rightx_current = -1, -1
    min_lane_distance = 80  # 차선 사이 최소 허용 거리 (픽셀)
   
 
    if len(contour_centers) >= 2:
        first = contour_centers[0]
        second = contour_centers[1]
        distance = abs(second - first)

        if distance > min_lane_distance:
            leftx_current, rightx_current = first, second
            prev_c0,prev_c1=first,second
           
            print(f"[contour 기반] left: {leftx_current}, right: {rightx_current}")
        else:
            # 너무 가까우면, 화면 위치 기준으로 하나만 선택
            if first < abs(second-first)//2:   #########################################################   line seperate part
                leftx_current = second
                prev_c0= second
                print("[contour 기반] 컨투어 너무 가까움 → 왼쪽만 유효")
            else:
                rightx_current = first
                prev_c1= first
                print("[contour 기반] 컨투어 너무 가까움 → 오른쪽만 유효")
                 
               
    elif len(contour_centers) == 1:
        if abs(contour_centers[0]-prev_c0) < abs(contour_centers[0]-prev_c1): ###############################################   line seperate part
            leftx_current = contour_centers[0]
            prev_c0=contour_centers[0]
            prev_c1=warp_img_w ######
            print("[contour 기반] 하나만 감지 - 왼쪽")
        else:
            rightx_current = contour_centers[0]
            prev_c0=0
            prev_c1=contour_centers[0]
            print("[contour 기반] 하나만 감지 - 오른쪽")

   



 #   contour_centers = sorted(contour_centers)
  #  leftx_current, rightx_current = -1, -1
   # min_lane_distance = 80  # 차선 사이 최소 허용 거리 (픽셀)
   
 
    #if len(contour_centers) >= 2:
     #   first = contour_centers[0]
      #  second = contour_centers[1]
       # distance = abs(second - first)

       # if distance > min_lane_distance:
        #    leftx_current, rightx_current = first, second
         #   print(f"[contour 기반] left: {leftx_current}, right: {rightx_current}")
        #else:
         #   # 너무 가까우면, 화면 위치 기준으로 하나만 선택
          #  if first < warp_img_w // 2:   #########################################################   line seperate part
           #     leftx_current = second
            #    print("[contour 기반] 컨투어 너무 가까움 → 왼쪽만 유효")
            #else:
             #   rightx_current = first
              #  print("[contour 기반] 컨투어 너무 가까움 → 오른쪽만 유효")
    #elif len(contour_centers) == 1:
     #   if contour_centers[0] < lane.shape[1] // 2: ###############################################   line seperate part
      #      leftx_current = contour_centers[0]
       #     print("[contour 기반] 하나만 감지 - 왼쪽")
        #else:
         #   rightx_current = contour_centers[0]
          #  print("[contour 기반] 하나만 감지 - 오른쪽")




  #  contour_centers = sorted(contour_centers)
   # leftx_current, rightx_current = -1, -1
   
    #if len(contour_centers) >= 2:
     #   leftx_current, rightx_current = contour_centers[:2]
      #  print(f"[contour 기반] left: {leftx_current}, right: {rightx_current}")
    #elif len(contour_centers) == 1:
     #   if contour_centers[0] < lane.shape[1] // 2:
      #      leftx_current = contour_centers[0]
       # else:
        #    rightx_current = contour_centers[0]

    # 7. Sliding window search
   
   
   
    window_height = int(lane.shape[0] / nwindows)
    nz = lane.nonzero()
    left_lane_inds = []
    right_lane_inds = []
    lx, ly, rx, ry = [], [], [], []

    out_img = np.dstack((lane, lane, lane)) * 255

    for window in range(nwindows):
        win_yl = lane.shape[0] - (window + 1) * window_height
        win_yh = lane.shape[0] - window * window_height

        if leftx_current != -1:
            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
           
            cv2.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2) # 위에서 구한 좌표 바탕으로 슬라이싱 윈도우 생성
           
            good_left_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) &
                              (nz[1] >= win_xll) & (nz[1] < win_xlh)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nz[1][good_left_inds]))
            lx.append(leftx_current)
            ly.append((win_yl + win_yh) / 2)

        if rightx_current != -1:
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin
           
            cv2.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2)
           
            good_right_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) &
                               (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]
            right_lane_inds.append(good_right_inds)
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nz[1][good_right_inds]))
            rx.append(rightx_current)
            ry.append((win_yl + win_yh) / 2)

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else []
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else []

    left_detected = len(left_lane_inds) > 600
    right_detected = len(right_lane_inds) > 600

    if left_detected:
        lfit = np.polyfit(np.array(ly), np.array(lx), 2)
    else:
        lfit = np.array([0, 0, 0])

    if right_detected:
        rfit = np.polyfit(np.array(ry), np.array(rx), 2)
    else:
        rfit = np.array([0, 0, 0])

    out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
    out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]

    cv2.imshow("05_out_img", out_img)

    return lfit, rfit, left_detected, right_detected






def draw_lane(image, warp_img, Minv, left_fit, right_fit):
   
    #if np.all(left_fit == 0) or np.all(right_fit == 0):
     #   return image
   

    yMax = warp_img.shape[0] # 버드아이뷰 변환한 이미지의 전체y좌표

   
    ploty = np.linspace(0, yMax - 1, yMax)  
    color_warp = np.zeros_like(warp_img).astype(np.uint8)  
   

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
   
     
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts = np.hstack((pts_left, pts_right)) # 위 두 배열을 수평으로 붙인다 --> 왼쪽차선 -> 오른쪽 차선으로 fillpoly이 따라가면서 다각형을 만들어줄 좌표들의 순서를 전달하기 위해 하나로 합침
 
    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))

    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)


def get_pos_from_fit(fit, y=warp_Offset, left=False, right=False):
   
    if np.all(fit==0): ####??????
        return 0
   
     
    else:
        a,b,c = fit #라인이 있으면 원근변환된 화면의 중간값인 warp_Offset에서의 x좌표를 구한다
        pos = a*y*y + b*y + c
        return pos



def line_point(frame):    # 버드아이로 볼 면적 확인용 함수
    #cv2.circle(frame,(90,320),5,(0,0,255),-1)          #(이미지,중심좌표,반지름,색상,두께)
    #cv2.circle(frame,(20,420),5,(0,0,255),-1)
    #cv2.circle(frame,(550,320),5,(0,0,255),-1)
    #cv2.circle(frame,(620,420),5,(0,0,255),-1)

    #cv2.circle(frame,(60,350),5,(0,0,255),-1)          #(이미지,중심좌표,반지름,색상,두께)
    #cv2.circle(frame,(10,450),5,(0,0,255),-1)
    #cv2.circle(frame,(580,350),5,(0,0,255),-1)
    #cv2.circle(frame,(630,450),5,(0,0,255),-1)

   
    #cv2.circle(frame,(60,400),5,(0,0,255),-1)          #(이미지,중심좌표,반지름,색상,두께)
    #cv2.circle(frame,(10,450),5,(0,0,255),-1)
    #cv2.circle(frame,(580,400),5,(0,0,255),-1)
    #cv2.circle(frame,(630,450),5,(0,0,255),-1)    ***********

    #cv2.circle(frame,(45,400),5,(0,0,255),-1)          #(이미지,중심좌표,반지름,색상,두께)
    #cv2.circle(frame,(10,450),5,(0,0,255),-1)
    #cv2.circle(frame,(595,400),5,(0,0,255),-1)               #gooooooodddddddddddddddddddddddddddd
    #cv2.circle(frame,(630,450),5,(0,0,255),-1)
   
   
    #cv2.circle(frame,(55,350),5,(0,0,255),-1)        
    #cv2.circle(frame,(10,450),5,(0,0,255),-1)
    #cv2.circle(frame,(595,350),5,(0,0,255),-1)
    #cv2.circle(frame,(630,450),5,(0,0,255),-1)                      ### fucking goooooood
   
    cv2.circle(frame,(150,330),5,(0,0,255),-1)        
    cv2.circle(frame,(46,379),5,(0,0,255),-1)
    cv2.circle(frame,(480,330),5,(0,0,255),-1)
    cv2.circle(frame,(596,379),5,(0,0,255),-1)        
   
    #cv2.circle(frame,(65,400),5,(0,0,255),-1)
    #cv2.circle(frame,(550,400),5,(0,0,255),-1)
   


    cv2.imshow("line_point",frame)


def draw_steer(lpos, rpos, left_fit, right_fit):
    global prev_steer_angle  # 전역 변수 사용

    line_center = (lpos + rpos) / 2
    real_center = warp_img_w / 2
    line_err = real_center - line_center
    err_trans = line_err * 4200/190  #2200 / 50     #4200/50   #1500 / 180
    y_val = 61#230



   
    center_k = 0.03
    center_correction = center_k * (real_center - line_center)



    # 곡률 계산
    if np.all(left_fit == 0):
        l_curve = np.inf
    else:
        l_curve = ((1 + (2 * left_fit[0] * y_val + left_fit[1]) ** 2) ** 1.5) / np.abs(2 * left_fit[0])

    if np.all(right_fit == 0):
        r_curve = np.inf
    else:
        r_curve = ((1 + (2 * right_fit[0] * y_val + right_fit[1]) ** 2) ** 1.5) / np.abs(2 * right_fit[0])

    # 곡률 평균 계산
    if np.isinf(l_curve) and not np.isinf(r_curve):
        curve_c = r_curve
    elif not np.isinf(l_curve) and np.isinf(r_curve):
        curve_c = l_curve
    elif not np.isinf(l_curve) and not np.isinf(r_curve):
        curve_c = (l_curve + r_curve) / 2
    else:
        curve_c = 1e6  # 매우 완만한 곡률 (직선처럼)

    #  원래 조향각 (라디안 → 도)
   
    steer_angle = np.arctan(err_trans / curve_c) * (180 / np.pi)
   
    steer_angle += center_correction

    #steer_angle_raw = np.arctan(err_trans / curve_c) * (180 / np.pi)
   
    #steer_angle_raw += center_correction

    #  조향각 보정: 이전 값과 섞어서 급변 억제
    #alpha = 0.6  # 값 클수록 이전 값 유지, 작을수록 새 값 반영
    #steer_angle = alpha * prev_steer_angle + (1 - alpha) * steer_angle_raw
    #prev_steer_angle = steer_angle

    # 시각화용 선 그리기
    line_length = 100
    end_x = int(320 - line_length * np.sin(np.radians(steer_angle)))
    end_y = int(480 - line_length * np.abs(np.cos(np.radians(steer_angle))))

    # 제한 범위 적용 (옵션)
    #if steer_angle >= 35 and steer_angle <50:
     #   steer_angle = 38
    #elif steer_angle <= -35 and steer_angle > -50:
     #   steer_angle = -38

    #if steer_angle >= 60:
     #   steer_angle = 60
       
    #elif steer_angle <= -60:
     #   steer_angle = -60        
     
    if steer_angle >= 5 :
        steer_angle = 5
       
    elif steer_angle <= -5:
        steer_angle = -5        
    return (steer_angle, end_x, end_y)



def start():
    global prev_steer_angle
    rospy.init_node('line_node')
   
    steer_pub = rospy.Publisher('/steer_info',Int32, queue_size=10)
    last_publish_time = time.time()  #time check
   
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
 
    print("start")

    while  cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
     
        cv2.imshow("original",frame) #-- 원본 이미지

        #line_point(frame)  # 버드아이뷰로 Q볼 지점 확인용 함수

        image = calibrate_image(frame)  
        #cv2.imshow("warp", image) #-- 왜곡 보정 이미지

        warp_img = warp_image(image)    
        #cv2.imshow("bird_view", warp_img) #버드아이뷰    ---    아직까지 딜레이 ㄱㅊ음
       
        left_fit, right_fit, left_detected, right_detected = warp_process_image(warp_img)
        lane_img = image#draw_lane(image, warp_img, Minv, left_fit, right_fit) ###############################################here is problommmmmmmmmmmmmmmmmmm
        #cv2.imshow("line_nemo", lane_img)  # 차선에 네모 그림
       
        lpos = get_pos_from_fit(left_fit, y=warp_Offset, left=True, right=False) #원본 화면상 y좌표 중간에서의 왼쪽촤선의 x좌표를 반환  
        rpos = get_pos_from_fit(right_fit, y=warp_Offset, left=False, right=True) #원본 화면상 y좌표 중간에서의 왼쪽촤선의 x좌표를 반환
       
       
        lane_width = 600#190#485
       
        if not left_detected and right_detected:
            lpos = rpos - lane_width
            print("left nono")  
       
         
        elif left_detected and not right_detected:
            rpos = lpos + lane_width
            print("right line nono")
           
           
           
        if not left_detected and not right_detected:
           steer_angle= prev_steer_angle
           end_x, end_y = 320,480            #************************************ here fix maybe or not
           print("both line nono")
           
        else:
            steer_angle,end_x,end_y=draw_steer(lpos,rpos,left_fit,right_fit)
            prev_steer_angle = steer_angle
       
       
       
        steer_angle,end_x,end_y = draw_steer(lpos,rpos,left_fit,right_fit)  
       
       
       

        cv2.line(lane_img, (320,480), (end_x,end_y),(255,0,0),2)
        cv2.putText(lane_img, f'steerangle: {steer_angle:.2f} deg' , (30,100),               cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        #cv2.imshow("fianl", lane_img)  # 차선에 네모 그림
       
       
        #steer_pub.publish(int(steer_angle))
        current_time = time.time()
        if current_time - last_publish_time >= 0.1: ### arduino send time
            steer_pub.publish(int(steer_angle))
            last_publish_time = current_time


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start()
