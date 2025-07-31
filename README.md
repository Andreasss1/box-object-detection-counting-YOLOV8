# 📦 Box Object Detection & Counting with YOLOv8

This project aims to detect and count boxes moving on a conveyor belt using the **YOLOv8** object detection model. It integrates detection, tracking, and both line & area-based counting for automated monitoring in industrial or logistics environments.

![Object Detection and Counting](https://github.com/Andreasss1/box-object-detection-counting-YOLOV8/blob/main/box-object-detection-counting.jpg?raw=true)

---

## 🚀 Key Features
- ✅ Detect boxes in video streams using YOLOv8
- 📍 Line Counting: Count boxes that cross a specific line
- 📦 Area Counting: Count boxes that enter a defined area
- 📹 Annotated output video with object count, IDs, and positions
- 📊 Video statistics and analysis (FPS, duration, etc.)

---

## 🧱 System Architecture

1. Extract ZIP files (video and labeled dataset)
2. Extract frames from the video
3. Split data into `train/` and `validation/` folders
4. Generate `data.yaml` configuration
5. Train YOLOv8 model on labeled data
6. Analyze video statistics
7. Run object detection and counting on video
8. Save and export the processed video

---

## 🧪 Sample Output

The output video displays:
- Bounding boxes for detected objects
- Unique IDs for each tracked object
- Movement trails
- Real-time count based on line and area crossings

---

## 🛠️ Technologies Used

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- Python 3
- Google Colab

---

<div align="center">

## 📬 Need a Similar Project? Let's Collaborate!
If you need a **custom IoT project** for **smart home, agriculture, industrial monitoring**, or other use cases,  
I’m ready to assist you!  

📧 **Reach out at:**  
### andreas.sebayang9999@gmail.com  

Let’s create something amazing together! 🚀

</div>
