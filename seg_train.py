from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(' ', task='segment')

    results = model.train(data='',
                          epochs=100,
                          batch=8,
                          imgsz=640,
                          device=0,
                          workers=4,
                          project='',
                          name='',
                          optimizer='SGD',
                          seed=0,
                          cos_lr=True,
                          close_mosaic=10,
                          lr0=0.001,
                          lrf=0.01,
                          val=True,
                          amp=True)
