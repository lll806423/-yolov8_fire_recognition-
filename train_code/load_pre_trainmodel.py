from ultralytics import YOLO

#加载模型
model = YOLO("yolo26s.pt")
results = model.predict(
    source="bus.jpg",
    save=True,
    name="predict",          # 固定文件夹名
    exist_ok=True             # 允许覆盖，不生成 predict2
)

print("✅ 模型加载成功，检测结果已保存在 runs/detect 文件夹里")

