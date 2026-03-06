## Lab 1

## Lab 2: YOLOv11n Object Detection Task

### No Datasets
Running the YOLO model without datasets.

![Pre-trained YOLOv11n 1](images/image2.png) ![Pre-trained YOLOv11n 2](images/image6.png) ![Pre-trained YOLOv11n 3](images/image8.png)

**Fig. 1. Pre-trained YOLOv11n Object Detection**

This figure shows the performance of the base model before custom training. While it successfully detects humans, it exhibits class confusion by misidentifying the oscilloscope as a microwave and the Jetson board as a keyboard or mouse because these specific hardware classes were not in the original dataset.

### Part 3

**Dataset10**

![Custom Detection Dataset-10 1](images/image11.png) ![Custom Detection Dataset-10 2](images/image4.png)

**Fig. 2. Custom Detection: Dataset-10**

Detection results using the smallest dataset, consisting of 10 images per class. According to the recorded data, this model achieved an mAP@50 of 0.995 but a lower mAP@50-95 of 0.74625, indicating that while it identifies the objects, the bounding box precision is still developing.

**Dataset25**

![Custom Detection Dataset-25 1](images/image10.png) ![Custom Detection Dataset-25 2](images/image7.png)

**Fig. 3. Custom Detection: Dataset-25**

Results using the mid-sized dataset. This model showed a significant improvement in reliability, reaching a Recall of 1.0 and a Precision of 0.98454. The bounding boxes are more stable, and the model correctly identifies the oscilloscope and Jetson even at varied angles.

**Dataset50**

![Custom Detection Dataset-50 1](images/image9.png) ![Custom Detection Dataset-50 2](images/image1.png) ![Custom Detection Dataset-50 3](images/image3.png) ![Custom Detection Dataset-50 4](images/image5.png)

**Fig. 4. Custom Detection: Dataset-50**

Inference using the largest dataset. This model provides the most robust results with an mAP@50-95 of 0.8751, the highest among all three tests. The increased data volume allowed the model to maintain high accuracy and stable confidence scores (averaging 0.89) despite background noise or changes in lighting.

### 5. Data & Observations

| Metric | Dataset_10 | Dataset_25 | Dataset_50 |
| :--- | :--- | :--- | :--- |
| **mAP@50** | 0.995 | 0.995 | 0.8951 |
| **mAP@50-95** | 0.74625 | 0.86874 | 0.8751 |
| **Precision** | 0.64866 | 0.98454 | 0.89151 |
| **Recall** | 0.82917 | 1.0 | 0.9 |
| **Inference Speed** | 15.4ms - 16.7ms | 21.0ms | 15.9ms |


## Lab 3: NanoOwl Vision Transformer


## Lab 4
