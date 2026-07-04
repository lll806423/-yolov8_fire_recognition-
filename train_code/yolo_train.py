import os
from ultralytics import YOLO
import attention_patch  # 注册 C2fPSA 模块（兼容不同 Ultralytics 版本）

# ========== 使用带注意力机制的自定义模型 ==========
# 基于 yolov8s + C2fPSA (位置-空间注意力) 在 backbone 深层
model = YOLO("yolov8s_attention.yaml")
model.load("runs/detect/yolov8_fire614-7/weights/best.pt")   # 加载已训练好的旧模型权重
print("注意力增强模型加载完成，开始训练...")
# ================================================


def train_model():

    model.train(                                                             
        data="/home/featurize/my_project/fire.v2i.yolov8.zip",

        epochs=130,     #100轮
        imgsz=1280,      #图片大小
        batch=24,        #每次送入的图片
        device="0",
        name="yolov8_fire_attn_newdata",
        patience=30,

        #学习率
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        #权重
        dfl=2.0,
        cls=0.5,
        box=7.5,

        #数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

    )

if __name__ == '__main__':
    train_model()
