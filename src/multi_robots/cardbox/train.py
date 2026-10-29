#!/usr/bin/env python3

from ultralytics import YOLO

def main():

    # Load pretrained model
    model = YOLO("yolov8s.pt")

    # Train
    results = model.train(
        data="/home/einfochips/Downloads/cardbox/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu",       
        workers=8,
        patience=20,
        cache=True,
        project="box_detector",
        name="yolov8s_box",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )

    print("Training Complete")
    print(results)

if __name__ == "__main__":
    main()
