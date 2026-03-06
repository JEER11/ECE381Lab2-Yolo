## Lab 1

## Lab 2: YOLOv11n Object Detection Task

### No Datasets
Running the YOLO model without datasets.
<img width="632" height="515" alt="image3" src="https://github.com/user-attachments/assets/083fd4db-f0d0-4d3a-9cf2-e100e7b36942" />
<img width="649" height="508" alt="image2" src="https://github.com/user-attachments/assets/f689215f-c4a6-4702-a9cb-e7f61d0ce62c" />
<img width="632" height="515" alt="image1" src="https://github.com/user-attachments/assets/36fbb01d-6cff-4a1c-b94b-6bc536f8dede" />
**Fig. 1. Pre-trained YOLOv11n Object Detection**

This figure shows the performance of the base model before custom training. While it successfully detects humans, it exhibits class confusion by misidentifying the oscilloscope as a microwave and the Jetson board as a keyboard or mouse because these specific hardware classes were not in the original dataset.

### Part 3

**Dataset10**

<img width="632" height="515" alt="image5" src="https://github.com/user-attachments/assets/ad830f33-1877-49b2-8eb3-ca93c6f570ef" />
<img width="642" height="520" alt="image4" src="https://github.com/user-attachments/assets/d02feae6-ce6d-46e5-8f80-a9e650b285fa" />
**Fig. 2. Custom Detection: Dataset-10**

Detection results using the smallest dataset, consisting of 10 images per class. According to the recorded data, this model achieved an mAP@50 of 0.995 but a lower mAP@50-95 of 0.74625, indicating that while it identifies the objects, the bounding box precision is still developing.

**Dataset25**

<img width="643" height="519" alt="image7" src="https://github.com/user-attachments/assets/2888ef8b-ec28-408b-b9d7-9d8fb8467287" />
<img width="649" height="508" alt="image6" src="https://github.com/user-attachments/assets/ed197a2a-df85-42e7-8005-1f16288f5b8d" />
**Fig. 3. Custom Detection: Dataset-25**

Results using the mid-sized dataset. This model showed a significant improvement in reliability, reaching a Recall of 1.0 and a Precision of 0.98454. The bounding boxes are more stable, and the model correctly identifies the oscilloscope and Jetson even at varied angles.

**Dataset50**

<img width="642" height="520" alt="image11" src="https://github.com/user-attachments/assets/ece0b1a0-cf1f-4376-82c8-22ba183580bf" />
<img width="643" height="519" alt="image10" src="https://github.com/user-attachments/assets/d022b1f4-c00b-4166-a4b9-1c0851f561de" />
<img width="632" height="515" alt="image9" src="https://github.com/user-attachments/assets/527083e4-159b-42be-9322-36b1af0699e9" />
<img width="638" height="517" alt="image8" src="https://github.com/user-attachments/assets/871e3a2d-d580-4122-ae3f-1a6938be02fe" />
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
