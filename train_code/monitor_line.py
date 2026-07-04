from ultralytics.utils.plotting import plot_results
from ultralytics import YOLO
import os
# 本地D盘下载的results.csv完整路径
#csv_file = r"runs/detect/yolov8_fire_attn_newdata-3/results.csv"

# 只传文件路径，图表会自动保存在csv同文件夹
#plot_results(csv_file)
#print("曲线图片 results.png 已生成在csv同级目录")


# 加载你下载到D盘的最优模型
from ultralytics import YOLO
import os

# 所有逻辑必须放进主函数判断块
if __name__ == '__main__':
    # ====================== 配置区 ======================
    weight_path = r"runs/detect/yolov8_fire_attn_newdata-3/weights/best.pt"
    data_yaml = r"D:\pycharm\pyproject\Machine_Learning\fire_recognition\fire.v2i.yolov8_621\data.yaml"
    save_target_dir = r"runs/detect/yolov8_fire_attn_newdata-3"
    # ====================================================

    os.makedirs(save_target_dir, exist_ok=True)
    model = YOLO(weight_path)

    metrics = model.val(
        data=data_yaml,
        plots=True,
        save_json=True,
        conf=0.25,
        workers=0,  # 关键：Windows强制单进程加载，彻底规避多进程报错
        project=os.path.dirname(save_target_dir),
        name=os.path.basename(save_target_dir),
        exist_ok=True
    )
    print(f"混淆矩阵、评估图表已保存至：{save_target_dir}")