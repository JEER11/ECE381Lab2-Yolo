# Lab 1

## Lab 2: YOLOv11n Object Detection Task

### No Datasets
Running the YOLO model without datasets.

<p align="center">
  <img width="32%" alt="image3" src="https://github.com/user-attachments/assets/083fd4db-f0d0-4d3a-9cf2-e100e7b36942" />
  <img width="32%" alt="image2" src="https://github.com/user-attachments/assets/f689215f-c4a6-4702-a9cb-e7f61d0ce62c" />
  <img width="32%" alt="image1" src="https://github.com/user-attachments/assets/36fbb01d-6cff-4a1c-b94b-6bc536f8dede" />
</p>

**Fig. 1. Pre-trained YOLOv11n Object Detection**

This figure shows the performance of the base model before custom training. While it successfully detects humans, it exhibits class confusion by misidentifying the oscilloscope as a microwave and the Jetson board as a keyboard or mouse because these specific hardware classes were not in the original dataset.

### Part 3

**Dataset10**

<p align="center">
  <img width="48%" alt="image5" src="https://github.com/user-attachments/assets/ad830f33-1877-49b2-8eb3-ca93c6f570ef" />
  <img width="48%" alt="image4" src="https://github.com/user-attachments/assets/d02feae6-ce6d-46e5-8f80-a9e650b285fa" />
</p>

**Fig. 2. Custom Detection: Dataset-10**

Detection results using the smallest dataset, consisting of 10 images per class. According to the recorded data, this model achieved an mAP@50 of 0.995 but a lower mAP@50-95 of 0.74625, indicating that while it identifies the objects, the bounding box precision is still developing.

**Dataset25**

<p align="center">
  <img width="48%" alt="image7" src="https://github.com/user-attachments/assets/2888ef8b-ec28-408b-b9d7-9d8fb8467287" />
  <img width="48%" alt="image6" src="https://github.com/user-attachments/assets/ed197a2a-df85-42e7-8005-1f16288f5b8d" />
</p>

**Fig. 3. Custom Detection: Dataset-25**

Results using the mid-sized dataset. This model showed a significant improvement in reliability, reaching a Recall of 1.0 and a Precision of 0.98454. The bounding boxes are more stable, and the model correctly identifies the oscilloscope and Jetson even at varied angles.

**Dataset50**

<p align="center">
  <img width="48%" alt="image11" src="https://github.com/user-attachments/assets/ece0b1a0-cf1f-4376-82c8-22ba183580bf" />
  <img width="48%" alt="image10" src="https://github.com/user-attachments/assets/d022b1f4-c00b-4166-a4b9-1c0851f561de" />
  <br>
  <img width="48%" alt="image9" src="https://github.com/user-attachments/assets/527083e4-159b-42be-9322-36b1af0699e9" />
  <img width="48%" alt="image8" src="https://github.com/user-attachments/assets/871e3a2d-d580-4122-ae3f-1a6938be02fe" />
</p>
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


# Lab 3: NanoOwl Vision Transformer

### 4. Procedure
We conducted a test using the web interface to observe the model's performance when adding multiple face characteristics. The model successfully detected all specified facial features without any issues.

<p align="center">
<img width="654" height="596" alt="image84" src="https://github.com/user-attachments/assets/21a8f445-e0a0-42d2-871d-b9f6b11df346" />
</p>

**Fig 1. Face, nose, eye, and mouth detection through a website**

### Part 1: Baseline & Initial Detection 
The initial phase involved establishing a baseline for the NanoOWL Vision Transformer utilizing both the web interface and terminal-based scripts.

<p align="center">
<img width="654" height="596" alt="image75" src="https://github.com/user-attachments/assets/868c91ce-3d48-4bd8-b2af-223c5f65f400" />
</p>

**Fig 2. Face detection through a website**

* **Hierarchical Detection (Fig 1 & 2):** We first utilized the `tree_demo` web interface to evaluate the model's nested detection capabilities. By inputting the prompt `[a face [a nose, an eye, a mouth]]`, we confirmed that the model successfully localized sub-features within a parent object without experiencing confusion.

<p align="center">
<img width="1277" height="428" alt="image80" src="https://github.com/user-attachments/assets/1b4f9ea8-f0e4-444f-a2fd-b92ca9b0db98" />
</p>

**Fig 3. Face Detection through the terminal**

* **Terminal Execution & Static Capture (Fig 3):** We executed the `attention_heatmap.py` script via the terminal to observe the model's focus on a single subject. The script successfully localized the face and provided a per-patch attention map that visualized the transformer's specific focus areas.

<p align="center">
<img width="1807" height="606" alt="image69" src="https://github.com/user-attachments/assets/bee84570-ae73-4131-a868-8512d05856ac" />
</p>

**Fig 4. Water Bottle Detection**

* **Object Generalization (Fig 4):** To test the model's zero-shot capabilities beyond human features, we tested its ability to detect a water bottle. Even when the bottle was oriented sideways and partially cut off by the frame, the model successfully maintained an accurate bounding box.

### Part 2: Prompt Engineering Experiments
In this section, we systematically varied the text prompts to interrogate how the Vision Transformer (ViT) architecture handles specificity, uncertainty, and emotional attributes.

#### Experiment A: Specificity Ladder
We observed that as the prompts became increasingly descriptive, the detection scores fluctuated. In several instances, the model's global attention shifted to background subjects rather than prioritizing the primary subject in the foreground. For prompts such as "a person" or "a face," the model occasionally prioritized individuals in the background, suggesting that the ViT's attention can be distracted by higher-entropy features elsewhere in the frame. In most cases with specific prompts, it recognized a person from behind rather than the subject directly in front of the camera.

| Prompt | Detection Score | Visual Results (Captured Image, Per-Patch Score, Detection + Heatmap) |
| :--- | :--- | :--- |
| "an object" | 0.01 | <img width="1703" height="548" alt="image76" src="https://github.com/user-attachments/assets/a84ac171-4c83-434e-a3cd-0c1faa983d3b" />
 |
| "a person" | 0.16 | <img width="1703" height="548" alt="image29" src="https://github.com/user-attachments/assets/83205856-64ff-452a-9ba4-c21caa8fdf80" />
 |
| "a face" | 0.36 | <img width="1703" height="548" alt="image57" src="https://github.com/user-attachments/assets/0d903f7a-9cae-4e45-a551-662c7769ec4f" />
 |
| "a human face with glasses" | 0.45 | <img width="1703" height="548" alt="image79" src="https://github.com/user-attachments/assets/a6403291-4f96-494d-8b52-c9d2d844f11b" />
 |
| "a male face with glasses and a beard" | 0.16 | <img <img width="1639" height="540" alt="image10" src="https://github.com/user-attachments/assets/6745dff2-7adb-4c09-94f2-f34e778b962b" />
 |

#### Experiment B: Wrong Prompts
When deliberately utilizing prompts for objects not present in the frame (e.g., "dog" or "car"), the model did not return a null result. Instead, it landed on background elements or anatomical features, such as a shoulder, attempting to find the closest visual approximation to the text embedding. This indicates that the model handles uncertainty by forcing a match rather than providing a low-confidence rejection.

| Prompt | Detection Score | Visual Results (Captured Image, Per-Patch Score, Detection + Heatmap) |
| :--- | :--- | :--- |
| "a dog" | 0.01 | <img src="images/combined_dog.jpg" width="100%"/> |
| "a car" | Not Recorded | <img src="images/combined_car.jpg" width="100%"/> |
| "a chair" | 0.07 | <img src="images/combined_chair.jpg" width="100%"/> |

#### Experiment C: Adversarial Prompts

| Prompt | Detection Score | Visual Results (Captured Image, Per-Patch Score, Detection + Heatmap) |
| :--- | :--- | :--- |
| "a face but not wearing glasses" | 0.21 | <img src="images/combined_no_glasses.jpg" width="100%"/> |
| "a happy face" | 0.50 | <img src="images/combined_happy.jpg" width="100%"/> |
| "a sad face" | 0.52 | <img src="images/combined_sad.jpg" width="100%"/> |

### Part 3: Tree Prompt Design & Failure Mode Documentation
The final phase involved designing complex hierarchies and documenting the physical conditions under which the model's performance degrades.

#### Activity A: Tree Prompt Refinement
We iterated through three versions of a tree prompt. While the model effectively detected "a man" and "standing," it struggled to identify a "water bottle" when nested within a larger hierarchical prompt. Even when the bottle was held close to the body or out to the side, the model failed to detect it within the tree structure, suggesting a limit to the complexity of nested objects it can process simultaneously.

| Iteration | Tree Prompt String | Visual Results (Detected & Missed) |
| :--- | :--- | :--- |
| 1 | `[A man]` | <img src="images/combined_tree_1.jpg" width="100%"/> |
| 2 | `[A man [standing]]` | <img src="images/combined_tree_2.jpg" width="100%"/> |
| 3 | `[A man [standing], [water bottle]]` | <img src="images/combined_tree_3.jpg" width="100%"/> |

## Lab 4


