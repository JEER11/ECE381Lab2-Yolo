# Lab 1: Image Classification & Regression

## 4. Procedure

### 4.1 Implementation Workflow

<p align="center">
  <img width="60%" alt="image43" src="https://github.com/user-attachments/assets/ac3e7720-8dca-45a4-a782-99926766db26" />
</p>

**Fig. 1. Code Workflow**

The system operates as a real-time pipeline on the Jetson hardware. The process initiates with the camera feed capturing raw frames, which subsequently undergo a pre-processing stage. This stage involves resizing the images to 224×224 pixels and applying normalization to ensure efficient data processing by the model backbone. Following pre-processing, the data is routed into the backbone for feature extraction. The pipeline concludes with either a classification output for discrete categories or a regression output for continuous coordinate tracking, depending on the specific task parameters.

### 4.2 Architecture
This experiment comprised a comparative analysis of four distinct neural network architectures to evaluate their efficacy in handling edge AI tasks on the Orin Nano platform.

### 4.3 Dataset Creation
The performance of each respective model was intrinsically linked to the quality and diversity of the captured training images. Custom datasets were generated for both classification and regression tasks utilizing interactive widgets. The primary objective during data collection was to introduce sufficient visual variety, enabling the models to generalize effectively rather than overfitting to a single spatial configuration.

#### 4.3.1 Image Classification (Thumbs Up & Thumbs Down)
Four architectures were evaluated: AlexNet, SqueezeNet, ResNet-18, and ResNet-34. The experimental procedure required sequential execution of code blocks to initialize both the camera feed and the data collection utility.

| AlexNet (Thumbs Down) | AlexNet (Thumbs Up) |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/cd1fccb2-4be8-47be-a0ef-5ef1fc635efa" width="100%"> | <img src="https://github.com/user-attachments/assets/2bb02aa4-fd09-4a7b-a5e4-d5c2f6c64a72" width="100%"> |
| **SqueezeNet (Thumbs Up)** | **SqueezeNet (Thumbs Up)** |
| <img src="https://github.com/user-attachments/assets/86ebd336-3ad3-4d5e-965f-211eec4d60a2" width="100%"> | <img src="https://github.com/user-attachments/assets/8c0403e1-b706-45ac-9191-c3c96dd8b60e" width="100%"> |

| ResNet-18 (Thumbs Down) | ResNet-18 (Thumbs Up) |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/4e71a541-889d-4e29-8c22-24a0e8314ae1" width="100%"> | <img src="https://github.com/user-attachments/assets/9a6f294e-8b35-4c1b-b74f-5f17810d988e" width="100%"> |
| **ResNet-34 (Thumbs Down)** | **ResNet-34 (Thumbs Up)** |
| <img src="https://github.com/user-attachments/assets/b18e23cd-b62f-4b47-b317-56aa0cdf3e3c" width="100%"> | <img width="1999" height="1114" alt="image10" src="https://github.com/user-attachments/assets/c6b38921-3ed3-42d0-8e8a-c4573d9c39a8" />
 |

Approximately 30 to 50 images were captured for each gesture category. To enhance model robustness, the dataset incorporated variations in angles, distances, and hands from multiple team members to ensure the network learned the generalized gesture rather than subject-specific features. During real-time inference, the interactive sliders provided immediate feedback, demonstrating high probability confidence for the correct gestures.

#### 4.3.2 Image Classification (Emotion Recognition)
This phase aimed to classify facial expressions into four specific states: None, Happy, Sad, and Angry.

| None | Happy |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/afaaa0f2-388e-44df-98d2-aeb15e6800a7" width="100%"> | <img src="https://github.com/user-attachments/assets/1dfe8a32-4a1a-4662-8fa5-0f80aeb589a4" width="100%"> |
| **Sad** | **Angry** |
| <img src="https://github.com/user-attachments/assets/4bb2d12a-4152-4d1f-b532-209bbd87dd0a" width="100%"> | <img src="https://github.com/user-attachments/assets/e497996b-9aca-4ea9-add2-7d58a28f4804" width="100%"> |

A balanced dataset consisting of 51 images per category was collected to train the ResNet-18 backbone. The model successfully distinguished between the emotional states by isolating key facial features. Live execution validated that the architecture could track and classify emotional shifts in real-time.

#### 4.3.3 Image Classification (Fingers 1-5)
This task introduced higher complexity by requiring the model to differentiate between five visually similar classes representing finger counts.

| Finger 1 | Finger 2 | Finger 3 |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/1832fb82-0f14-4980-b600-fbc7a99c2aa7" width="100%"> | <img src="https://github.com/user-attachments/assets/47773702-fd85-4c1f-b86b-af6b5cf1ce8c" width="100%"> | <img src="https://github.com/user-attachments/assets/e26f72be-6c28-4a97-99ed-a56600955faf" width="100%"> |
| **Finger 4** | **Finger 5** | |
| <img src="https://github.com/user-attachments/assets/d2b84753-406f-4088-b1bc-770157dcf601" width="100%"> | <img src="https://github.com/user-attachments/assets/dea62bf0-d2be-4fd7-923c-ad34000efc4c" width="100%"> | |

A larger dataset ranging from 111 to 162 images per class was collected to mitigate confusion between the structurally similar hand configurations. The ResNet-18 architecture achieved high accuracy, verifying its capability to process multiple discrete classifications simultaneously.

#### 4.3.4 Image Classification (Custom Gestures: Rock, Luck, Dog)
The final classification assessment utilized custom gestures.

| Rock | Luck | Dog |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/3100ebfc-9be5-43d5-ba1b-8603ac1307e7" width="100%"> | <img src="https://github.com/user-attachments/assets/b0f008f1-f65a-4518-a917-29102e2c2798" width="100%"> | <img width="1069" height="866" alt="image21" src="https://github.com/user-attachments/assets/f885b1cf-302a-479e-84ab-d3f04107b148" />
 |

A restricted dataset of 36 images per category was utilized for this trial. Despite the limited sample size, the model achieved 100% training accuracy. This outcome highlighted a clear instance of overfitting, as the network perfectly memorized the specific gestures within the static laboratory environment rather than learning generalized features.

#### 4.3.5 Image Regression (Nose, Left/Right Eye)
The regression task required the system to continuously track and output coordinates for specific facial keypoints: the nose, left eye, and right eye.

**ResNet-18 Regression Tracking**

| Nose | Left Eye | Right Eye |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/b34e4865-4c43-4a05-a008-d133fd42d7ea" width="100%"> | <img src="https://github.com/user-attachments/assets/f870923e-a4f0-42e9-9f4e-4be41dd50865" width="100%"> | <img src="https://github.com/user-attachments/assets/e01a065f-cf83-4dba-9f8f-dec1aa1366b5" width="100%"> |

**ResNet-34 Regression Tracking**

| Nose | Left Eye | Right Eye |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/fd4253c5-d29e-48ed-9527-e018e1c8bfd0" width="100%"> | <img src="https://github.com/user-attachments/assets/579c6b45-3709-4c2d-928b-95bad4de3179" width="100%"> | <img src="https://github.com/user-attachments/assets/80e73d25-a58c-48ae-8782-47396b37f3a5" width="100%"> |


## 5. Data & Observations
The recorded data illustrates the performance metrics derived from the interactive testing across varying architectures and epoch configurations.

* **Underfitting (30 Epochs):** The model exhibited elevated loss values and reduced accuracy, which translated to erratic and jittery slider responses during real-time testing.
* **Optimal Training (50 Epochs):** This hyperparameter setting yielded a robust balance, achieving 98.10% accuracy while maintaining the model's capacity to generalize to novel hand positions.
* **Overfitting (100 Epochs):** Although the training accuracy neared 100%, the model became excessively sensitized to the specific lighting conditions and background environments present during training.

### 5.1 Epoch Comparison
The following table compares the four required models for the Thumbs Up/Down classification task based on the final laboratory observations.

| Architecture | Dataset Count | Final Loss | Final Accuracy |
| :--- | :--- | :--- | :--- |
| **ResNet-18** | 334 | 0.0103 | 98.10% |
| **ResNet-34** | 395 | 0.0154 | 96.30% |
| **AlexNet** | 233 | 0.0876 | 47.39% |
| **SqueezeNet** | 303 | 0.0871 | 51.83% |


# Lab 2: YOLOv11n Object Detection Task

### No Datasets
Running the YOLO model without datasets.

<p align="center">
<img width="649" height="508" alt="image2" src="https://github.com/user-attachments/assets/0e80bdb5-33d8-4dc4-b327-193a03232f8e" />
<img width="649" height="508" alt="image6" src="https://github.com/user-attachments/assets/2a14a3e7-8cda-4bc2-bc8f-8ae680867d44" />
<img width="638" height="517" alt="image8" src="https://github.com/user-attachments/assets/53b5e2ce-91b9-4dd3-a0de-e89e4498c3e7" />
</p>

**Fig. 1. Pre-trained YOLOv11n Object Detection**

This figure shows the performance of the base model before custom training. While it successfully detects humans, it exhibits class confusion by misidentifying the oscilloscope as a microwave and the Jetson board as a keyboard or mouse because these specific hardware classes were not in the original dataset.

### Part 3

**Dataset10**

<p align="center">
<img width="642" height="520" alt="image11" src="https://github.com/user-attachments/assets/05035b07-dca5-4f1d-a36e-96d53601205c" />
<img width="642" height="520" alt="image4" src="https://github.com/user-attachments/assets/c58ed52d-aa8a-48c5-952d-a0362bf2c45e" />
</p>

**Fig. 2. Custom Detection: Dataset-10**

Detection results using the smallest dataset, consisting of 10 images per class. According to the recorded data, this model achieved an mAP@50 of 0.995 but a lower mAP@50-95 of 0.74625, indicating that while it identifies the objects, the bounding box precision is still developing.

**Dataset25**

<p align="center">
  <img width="48%" alt="image6" src="https://github.com/user-attachments/assets/ed197a2a-df85-42e7-8005-1f16288f5b8d" />
  <img width="643" height="519" alt="image7" src="https://github.com/user-attachments/assets/feb9f11a-b7dc-4564-83ba-a1c12173e68f" />

</p>

**Fig. 3. Custom Detection: Dataset-25**

Results using the mid-sized dataset. This model showed a significant improvement in reliability, reaching a Recall of 1.0 and a Precision of 0.98454. The bounding boxes are more stable, and the model correctly identifies the oscilloscope and Jetson even at varied angles.

**Dataset50**

<p align="center">
<img width="632" height="515" alt="image9" src="https://github.com/user-attachments/assets/649711b6-cf1e-4c68-a56e-f419eb9060b2" />
<img width="632" height="515" alt="image1" src="https://github.com/user-attachments/assets/8d6f845b-b29d-41db-bd0d-46c34571d542" />
  <br>
<img width="632" height="515" alt="image3" src="https://github.com/user-attachments/assets/067e7f9a-7f38-4d7b-8d58-bc9175574954" />
<img width="632" height="515" alt="image5" src="https://github.com/user-attachments/assets/c1e0d8ee-8dce-4591-b406-aa80628be7d9" />
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
| "an object" | 0.01 | <img width="1703" height="548" alt="image76" src="https://github.com/user-attachments/assets/a84ac171-4c83-434e-a3cd-0c1faa983d3b" /> |
| "a person" | 0.16 | <img width="1703" height="548" alt="image29" src="https://github.com/user-attachments/assets/83205856-64ff-452a-9ba4-c21caa8fdf80" /> |
| "a face" | 0.36 | <img width="1703" height="548" alt="image57" src="https://github.com/user-attachments/assets/0d903f7a-9cae-4e45-a551-662c7769ec4f" /> |
| "a human face with glasses" | 0.45 | <img width="1703" height="548" alt="image79" src="https://github.com/user-attachments/assets/a6403291-4f96-494d-8b52-c9d2d844f11b" /> |
| "a male face with glasses and a beard" | 0.16 | <img width="1639" height="540" alt="image10" src="https://github.com/user-attachments/assets/6745dff2-7adb-4c09-94f2-f34e778b962b" /> |

#### Experiment B: Wrong Prompts
When deliberately utilizing prompts for objects not present in the frame (e.g., "dog" or "car"), the model did not return a null result. Instead, it landed on background elements or anatomical features, such as a shoulder, attempting to find the closest visual approximation to the text embedding. This indicates that the model handles uncertainty by forcing a match rather than providing a low-confidence rejection.

| Prompt | Detection Score | Visual Results (Captured Image, Per-Patch Score, Detection + Heatmap) |
| :--- | :--- | :--- |
| "a dog" | 0.01 | <img width="1703" height="548" alt="image70" src="https://github.com/user-attachments/assets/1c63bdda-284e-42df-b2df-a63c266462e3" /> |
| "a car" | Not Recorded | <img width="1639" height="540" alt="image71" src="https://github.com/user-attachments/assets/2d3d0ea3-f608-42ff-89fb-135f4fcda061" /> |
| "a chair" | 0.07 | <img width="1639" height="540" alt="image14" src="https://github.com/user-attachments/assets/f6d4a49f-8dbb-426b-b20a-a3f1a2790066" /> |

#### Experiment C: Adversarial Prompts

| Prompt | Detection Score | Visual Results (Captured Image, Per-Patch Score, Detection + Heatmap) |
| :--- | :--- | :--- |
| "a face but not wearing glasses" | 0.21 | <img width="1639" height="540" alt="image25" src="https://github.com/user-attachments/assets/001117a3-2d5a-4c33-8707-968d01149cdb" /> |
| "a happy face" | 0.50 | <img width="1639" height="540" alt="image56" src="https://github.com/user-attachments/assets/6bcec686-e559-476e-ab2a-17ec254ef655" /> |
| "a sad face" | 0.52 | <img width="1639" height="540" alt="image11" src="https://github.com/user-attachments/assets/338ba6ec-65e0-4890-b8da-2fb26acc1a4f" /> |

### Part 3: Tree Prompt Design & Failure Mode Documentation
The final phase involved designing complex hierarchies and documenting the physical conditions under which the model's performance degrades.

#### Activity A: Tree Prompt Refinement
We iterated through three versions of a tree prompt. While the model effectively detected "a man" and "standing," it struggled to identify a "water bottle" when nested within a larger hierarchical prompt. Even when the bottle was held close to the body or out to the side, the model failed to detect it within the tree structure, suggesting a limit to the complexity of nested objects it can process simultaneously.

| Iteration | Tree Prompt String | Visual Results (Detected & Missed) |
| :--- | :--- | :--- |
| 1 | `[A man]` | <img width="1639" height="540" alt="image30" src="https://github.com/user-attachments/assets/df3eca8d-9113-4fbc-982f-978236ab81d1" /> |
| 2 | `[A man [standing]]` | <img width="1639" height="540" alt="image38" src="https://github.com/user-attachments/assets/b4c14b79-d3c3-4495-b0f0-b2c9edb0c51a" /> |
| 3 | `[A man [standing], [water bottle]]` | <img width="1639" height="540" alt="image62" src="https://github.com/user-attachments/assets/ebe820ba-241f-47d4-a3b7-5a6ea81ba5b9" /> |

#### Activity B: Failure Mode Analysis
Nine tests were conducted to identify the model's breaking points.

* **Distance Observations:** A notable drop in confidence occurred at a 1-meter distance with a score of 0.07, though the score recovered significantly at 3 meters to 0.45. This non-linear performance suggests sensitivity to the scale of the object patches.
* **Robustness:** The model demonstrated high resilience to occlusion and lighting. It successfully maintained face detection even when half the face was covered or when a phone flashlight was pointed directly at the lens.

| Test | Condition | Detection Score | Visual Results | Hypothesis for why it failed |
| :--- | :--- | :--- | :--- | :--- |
| **Occlusion** | Cover half your face with your hand | 0.39 | <img width="1639" height="540" alt="image21" src="https://github.com/user-attachments/assets/54ad8b50-dd11-4dcd-a3f3-cdc04d4e897b" /> | Didn't fail |
| **Lighting** | Point the phone flashlight directly at the camera | 0.42 | <img width="1639" height="540" alt="image64" src="https://github.com/user-attachments/assets/65196f7d-2747-488c-bd29-704b7ea1412d" /> | Didn't fail |
| **Lighting** | Dim the room lights as much as possible | Not Recorded | <img width="1639" height="540" alt="image53" src="https://github.com/user-attachments/assets/ecb35065-6b3b-487e-a2d1-98082c7eeda4" /> | Didn't fail |
| **Distance** | Sit as close as possible to the camera | Not Recorded | <img width="1639" height="540" alt="image81" src="https://github.com/user-attachments/assets/2a54703a-d58c-4efc-8832-f24979f4f6cf" /> | Didn't fail |
| **Distance** | Sit at a medium distance (~1m) | 0.07 | <img width="1639" height="540" alt="image17" src="https://github.com/user-attachments/assets/a4ab51f6-9efe-451a-b76a-f1eb385ece0e" /> | Didn't fail |
| **Distance** | Sit far from the camera (~3m) | 0.46 | <img width="1639" height="540" alt="image59" src="https://github.com/user-attachments/assets/ed14852a-0a85-4416-b363-e65dc56515ee" /> | Didn't fail |
| **Multi-person** | Two students in frame | 0.74 | <img width="1639" height="540" alt="image49" src="https://github.com/user-attachments/assets/5e5b28a1-0147-45db-ab05-b53965603d9b" /> | Didn't fail |
| **Rotation** | Tilt head 45 degrees | 0.31 | <img width="1639" height="540" alt="image33" src="https://github.com/user-attachments/assets/9395e288-11c2-4909-b4ad-0c83dc50f2e9" /> | Didn't fail |
| **Rotation** | Tilt head 90 degrees | 0.48 | <img width="1639" height="540" alt="image47" src="https://github.com/user-attachments/assets/e82f74f7-8fc9-42d1-88ff-dfd98d0ceda8" /> | Didn't fail |

# Lab 4
