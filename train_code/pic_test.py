import cv2
import os
import re
from ultralytics import YOLO

# ========== 配置参数 ==========
img_path = "material/fire.124.webp"        # 测试图片路径
model1_path = "runs/detect/yolov8_fire614-7/weights/best.pt"   # 第一个模型路径
model2_path = "runs/detect/yolov8_fire_attn_newdata-3/weights/best.pt"      # 第二个模型路径
conf = 0.1                              # 置信度阈值
# =============================
 
# ========== 输出目录：桌面"机器学习效果图" ==========
output_dir = r"C:\Users\19048\Desktop\test"
os.makedirs(output_dir, exist_ok=True)

# 从文件名提取编号（如 fire.49.png → 49）
match = re.search(r'(\d+)', os.path.basename(img_path))
img_index = match.group(1) if match else "0"
# ==============================================

# 定义颜色：模型1红色，模型2蓝色 (BGR)
color1 = (0, 0, 255)
color2 = (255, 0, 0)

# ------------------------------------------------
# 第一步：加载并运行模型1，单独显示
# ------------------------------------------------
print("=" * 40)
print("输出目录:", output_dir)
print("运行模型1...")
image1 = cv2.imread(img_path)
if image1 is None:
    print("图片读取失败，请检查路径！")
    exit()

model1 = YOLO(model1_path)
res1 = model1(image1, conf=conf)
for box in res1[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    score = float(box.conf[0])
    label = f"Model3 {score:.2f}"
    cv2.rectangle(image1, (x1, y1), (x2, y2), color1, 2)
    cv2.putText(image1, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2)

cv2.namedWindow("Model 1 - Fire Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Model 1 - Fire Detection", image1)
cv2.imwrite(os.path.join(output_dir, f"{img_index}_model1.jpg"), image1)
print(f"已保存到桌面：{img_index}_model1.jpg")
cv2.waitKey(1)  # 刷新模型1窗口
print("运行模型2...")
image2 = cv2.imread(img_path)
if image2 is None:
    print("图片读取失败，请检查路径！")
    exit()

model2 = YOLO(model2_path)
res2 = model2(image2, conf=conf)
for box in res2[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    score = float(box.conf[0])
    label = f"Model2 {score:.2f}"
    cv2.rectangle(image2, (x1, y1), (x2, y2), color2, 2)
    cv2.putText(image2, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

cv2.namedWindow("Model 2 - Fire Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Model 2 - Fire Detection", image2)
cv2.imwrite(os.path.join(output_dir, f"{img_index}_model2.jpg"), image2)
print(f"已保存到桌面：{img_index}_model2.jpg")
print("=" * 40)
print("全部完成！")
print(f"输出目录：{output_dir}")
