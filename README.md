# PA_Trustworthy_AI

An implementation for exploring trustworthy-AI concerns using Saliency Verbalization (YOLO detection + D-RISE / D-MFPP + Llama/LLM).
This repository accompanies the project report and in the folder `pa_trustworthy_ai_latex/`.

---

## Repository Structure

```
│ 
├── diagrams/                # Architecture / workflow diagrams
├── interviewTranscripts/    # Transcripts from interviews (qualitative insights)
├── pa_trustworthy_ai_latex/ # LaTeX source for the project write-up
├── src/
│   ├── driseYoloLlama.py    # Main program file (YOLO + LLM + D-RISE / D-MFPP pipeline)
│   └── ...                  # Other modules
├── pyproject.toml           # Poetry project config
├── poetry.lock              # Dependency lock file
└── README.md
```

---

## Setup

```bash
git clone https://github.com/rieselin/PA_Trustworthy_AI.git
cd PA_Trustworthy_AI

python -m venv ./.venv
source ./.venv/bin/activate     # Linux/Mac
# .venv\Scripts\activate        # Windows

poetry install
```

Ensure you have the required Python version (see `pyproject.toml`) and any GPU dependencies if running YOLO.

---

## Main Script: `driseYoloLlama.py`

The main script implements the pipeline:

1. **Run YOLO** on an input image
2. **Prepare the prompt** for the LLM
3. **Run Llama/LLM** for reasoning or trust-related assessment
4. **Apply Saliency** (D-RISE / D-MFPP based)
5. **Save results and meta of inputs**

---

## Example Usage

### Basic run

```bash
python src/driseYoloLlama.py 
```

### Arguments in driseYoloLlama.py Explained

| Argument                             | Description                                                   |
| ------------------------------------ | ------------------------------------------------------------- |
| `--img_names`                        | Names of the imagea used for YOLO + LLM reasoning             |
| `--yolo_model`                       | Path to the YOLO weights                                      |
| `--mask_type`                        | Type of mask to use for saliency: mfpp / rise                 |
| `--target_classes`                   | Classes in annotations that are to be processed for the input |
| `--instruction`                      | Instruction that is sent to Llama along with the image(s)     |
| `--run_only_first_bbox`              | Set to True for testing only first bbox                       |
| `--send_saliency_map`                | Set to True to send Saliency Map to Llama                     |
| `--send_labelled_bbox`               | Set to True to send image with labels to Llama                |
| `--send_predicted_bbox`              | Set to True to send Predicted Bbox to Llama                   |
| `--send_all_bboxes_of_image_at_once` | Set to True to send all Bboxes to Llama at once               |

---

## 📦 Example Meta Output (JSON)

```json
{
    "date_time_tag": "2025-11-13_11-04-11",
    "image_name": "000019",
    "model_path": "train13/weights/best.pt",
    "target_classes": [
        1,
        2,
        3,
        4,
        0
    ],
    "run_id_tag": "",
    "instruction": "describe why the object detection model made the bounding box prediciton based on the saliency map and the predicted bounding box on the image. \n    if there is no bounding box on the image, explain based on the bounding box why the model did not detect the object.\n    the saliency map colors are from blue (negative contribution) over green 0.9 (no contribution) to red (positive contribution)\n    do not explain any of the concepts in the instruction. \n    keep the answer concise within 100 words.",
    "send_to_llama": {
        "send_saliency_map": false,
        "send_labelled_bbox": false,
        "send_predicted_bbox": true
    },
    "predicted_bboxes": [
        {
            "bbox": [
                [
                    512,
                    233
                ],
                [
                    656,
                    233
                ],
                [
                    656,
                    398
                ],
                [
                    512,
                    398
                ]
            ],
            "confidence": 0.7192946672439575,
            "class_id": 0,
            "label": "Car"
        }
    ],
    "output_paths": {
        "output_root": "output/2025-11-13_11-04-11/000019/",
        "saliency_dir": "output/2025-11-13_11-04-11/000019/saliency/",
        "llama_dir": "output/2025-11-13_11-04-11/000019/llama/"
    },
    "composed_images": [
        "output/2025-11-13_11-04-11/000019/saliency/bbox_0_tc_0.png"
    ],
    "llama_responses": [
        "output/2025-11-13_11-04-11/000019/llama/llama_response0.txt"
    ]
}
```

---

## Where to Find What

* **Main program / pipeline logic**: `src/driseYoloLlama.py`
* **Other code files**: all inside `src/`
* **LaTeX report**: `pa_trustworthy_ai_latex/`
* **Diagrams & visual explanations**: `diagrams/`
* **Interview transcripts**: `interviewTranscripts/`
* **Dependencies**: `pyproject.toml` + `poetry.lock`

---

## Extending the Project

You can easily customize or expand:

* Swap YOLO model variants (simply change `--model_path`)
* Use a different mask type (`--mask_type`)

---

## Contributing

Feel free to fork the repository or submit issues / PRs with improvements.
