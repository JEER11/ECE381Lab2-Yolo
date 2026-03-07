## Lab 1

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


## Lab 3: NanoOwl Vision Transformer

### 4. Procedure
We conducted a test using the web interface to observe the model's performance when adding multiple face characteristics. The model successfully detected all specified facial features without any issues.

<p align="center">
<img width="654" height="596" alt="image84" src="https://github.com/user-attachments/assets/21a8f445-e0a0-42d2-871d-b9f6b11df346" />
</p>

**Fig 1. Face, nose, eye, and mouth detection through a website**

### Part 1: Baseline & Initial Detection The initial phase involved establishing a baseline for the NanoOWL Vision Transformer utilizing both the web interface and terminal-based scripts.

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

| Prompt | Detection Score |
| :--- | :--- |
| "an object" | 0.01 |
| "a person" | 0.16 |
| "a face" | 0.36 |
| "a human face with glasses" | 0.45 |
| "a male face with glasses and a beard" | 0.16 |

#### Experiment B: Wrong Prompts
When deliberately utilizing prompts for objects not present in the frame (e.g., "dog" or "car"), the model did not return a null result. Instead, it landed on background elements or anatomical features, such as a shoulder, attempting to find the closest visual approximation to the text embedding. This indicates that the model handles uncertainty by forcing a match rather than providing a low-confidence rejection.

| Prompt | Detection Score |
| :--- | :--- |
| "a dog" | 0.01 |
| "a car" | Not Recorded |
| "a chair" | 0.07 |

#### Experiment C: Adversarial Prompts

| Prompt | Detection Score |
| :--- | :--- |
| "a face but not wearing glasses" | 0.21 |
| "a happy face" | 0.50 |
| "a sad face" | 0.52 |

### Part 3: Tree Prompt Design & Failure Mode Documentation
The final phase involved designing complex hierarchies and documenting the physical conditions under which the model's performance degrades.

#### Activity A: Tree Prompt Refinement
We iterated through three versions of a tree prompt. While the model effectively detected "a man" and "standing," it struggled to identify a "water bottle" when nested within a larger hierarchical prompt. Even when the bottle was held close to the body or out to the side, the model failed to detect it within the tree structure, suggesting a limit to the complexity of nested objects it can process simultaneously.

| Iteration | Tree Prompt String |
| :--- | :--- |
| 1 | `[A man]` |
| 2 | `[A man [standing]]` |
| 3 | `[A man [standing], [water bottle]]` |

#### Activity B: Failure Mode Analysis
Nine tests were conducted to identify the model's breaking points.

* **Distance Observations:** A notable drop in confidence occurred at a 1-meter distance (score: 0.07), though the score recovered significantly at 3 meters (0.45). This non-linear performance suggests sensitivity to the scale of the object patches.
* **Robustness:** Surprisingly, the model demonstrated high resilience to Occlusion and Lighting. It successfully maintained face detection even when half the face was covered or when a phone flashlight was pointed directly at the lens.

| Test | Condition | Detection Score | Result |
| :--- | :--- | :--- | :--- |
| **Occlusion** | Cover half your face with your hand | 0.39 | Didn't fail |
| **Lighting** | Point the phone flashlight directly at the camera | 0.42 | Didn't fail |
| **Lighting** | Dim the room lights as much as possible | Not Recorded | Didn't fail |
| **Distance** | Sit as close as possible to the camera | Not Recorded | Didn't fail |
| **Distance** | Sit at a medium distance (~1m) | 0.07 | Didn't fail |
| **Distance** | Sit far from the camera (~3m) | Not Recorded | Didn't fail |
| **Multi-person** | Two students in frame | 0.74 | Didn't fail |
| **Rotation** | Tilt head 45 degrees | 0.31 | Didn't fail |
| **Rotation** | Tilt head 90 degrees | 0.48 | Didn't fail |

## Lab 4


