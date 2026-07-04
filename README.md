# Autonomous Robot Navigation System

<p align="center">
  <img src="https://img.shields.io/badge/PLATFORM-JETSON%20%7C%20REALSENSE-25344F?style=for-the-badge&labelColor=555555" alt="Platform: Jetson and RealSense" />
  <img src="https://img.shields.io/badge/SOFTWARE-PYTHON%20%2B%20ROS-0C8D7B?style=for-the-badge&labelColor=555555" alt="Software: Python and ROS" />
  <img src="https://img.shields.io/badge/PROJECT-AUTONOMOUS%20ROBOT-C65D00?style=for-the-badge&labelColor=555555" alt="Project: Autonomous Robot" />
  <img src="https://img.shields.io/badge/VISION-OPENCV%20%2B%20YOLOV4--TINY-1F5FDB?style=for-the-badge&labelColor=555555" alt="Vision: OpenCV and YOLOv4-tiny" />
</p>

This repository contains the source code for a ROS-based autonomous navigation system for a mobile robot. The system integrates two primary capabilities: vision-based lane following and depth-aware person tracking.
---
## System Architecture
The system consists of two main ROS nodes running in parallel:
1.  **Lane Detection Node (`lane_follower.py`)**: Processes a standard camera feed to identify lane lines and calculate the required steering angle to stay centered.
2.  **Person Tracking Node (`person_tracker.py`)**: Uses a YOLOv4-tiny model and an Intel RealSense depth camera to detect people, measure their distance, and issue high-level movement commands.
A simple diagram of the data flow is as follows:
```
[2D Camera (GStreamer)] --> [Lane Detection Node] ----> /steer_info ----> [Motor Controller]
                                                                                ^
[RealSense (Color+Depth)] --> [Person Tracking Node] ---> /bt_cmd ----------> [Motor Controller]
```
---
## Features
### 1. Lane Detection & Keeping
- **Perspective Transformation**: Converts the camera's front-facing view into a top-down "bird's-eye view" for stable lane detection.
- **Advanced Image Pre-processing**: Utilizes a Homomorphic filter to normalize brightness and contrast, improving robustness in varied lighting conditions.
- **Robust Lane Finding**: Employs a sliding window search algorithm combined with contour detection and second-degree polynomial fitting to accurately model lane curvature.
- **Real-time Steering Control**: Calculates a precise steering angle based on lane center deviation and curvature, publishing it to the `/steer_info` ROS topic for the motor controller.
### 2. Person Detection & Tracking
- **Real-time Object Detection**: Uses a GPU-accelerated YOLOv4-tiny model to detect people in the robot's path.
- **Accurate Depth Perception**: Leverages an Intel RealSense camera to get the precise distance to each detected person in meters.
- **Intelligent Target Following**: Identifies the nearest person and makes decisions based on their position and distance relative to the robot.
- **Smooth & Safe Commands**: Implements a time-based activation delay (e.g., 2 seconds) to ensure commands are issued only when a condition is stable, preventing jerky or erratic robot behavior.
- **High-Level Command Publishing**: Issues simple commands (`up`, `down`, `left`, `right`) to the `/bt_cmd` ROS topic, likely for a Bluetooth-connected microcontroller.
---
## Prerequisites
### Hardware
- An NVIDIA Jetson-like device (for CUDA acceleration).
- A standard 2D camera compatible with GStreamer (`nvarguscamerasrc`).
- An Intel RealSense Depth Camera (e.g., D435).
- A mobile robot platform with a motor controller subscribed to ROS topics.
### Software & Libraries
- Robot Operating System (ROS)
- Python 3
- OpenCV (`opencv-python`)
- NumPy
- PyRealSense2 (`pyrealsense2`)
### AI Models
- YOLOv4-tiny weights (`yolov4-tiny.weights`)
- YOLOv4-tiny config (`yolov4-tiny.cfg`)
- COCO class names (`coco.names`)
---
## How to Run
1.  **Setup the Environment:**
    - Ensure all prerequisite libraries and hardware are installed and connected.
    - Create a ROS workspace and a package.
    - Place the script files (renamed, e.g., `lane_follower.py` and `person_tracker.py`) in your package's `scripts` directory.
    - Place the YOLO model files in a designated folder (e.g., `~/yolot/`).
2.  **Make Scripts Executable:**
    ```bash
    chmod +x lane_follower.py
    chmod +x person_tracker.py
    ```
3.  **Launch the Nodes:**
    Open separate terminals or use a ROS launch file to run the nodes.
    ```bash
    # Terminal 1
    rosrun your_package_name lane_follower.py
    # Terminal 2
    rosrun your_package_name person_tracker.py
    ```
4.  **Implement the Controller:**
    You must create a separate ROS node that subscribes to the `/steer_info` (Int32) and `/bt_cmd` (String) topics. This node will be responsible for translating the published angles and commands into actual motor signals for your specific robot hardware.
