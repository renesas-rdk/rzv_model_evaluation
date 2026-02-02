# rzv_model_evaluation

A C++ package for evaluating DRP-AI compiled models on Renesas RZ/V2H hardware. This package provides tools to calculate standard object detection metrics (mAP50, mAP50-95) by comparing model inference results against ground truth datasets (YOLO/Roboflow format).

## Features

- **DRP-AI Integration:** Runs inference directly using the `rzv_model` and `rzv_yolov8` libraries.
- **Standard Metrics:** Calculates Precision, Recall, F1 Score, mAP@0.5, and mAP@0.5:0.95.
- **Dataset Support:** Supports standard YOLO/Roboflow dataset directory structures defined via YAML.
- **Extensible Architecture:** Designed with a base `ModelEvaluator` class to easily support new model types beyond YOLOv8.

## Prerequisites

- **Hardware:** RZ/V2H Robotics Development Kit
- **ROS 2 Distribution:** Jazzy or later
- **Dependencies:** see [package.xml](package.xml)

## Usage

The package provides an executable `yolov8_evaluation_runner` to evaluate a compiled YOLOv8 model against a test dataset.

### Dataset Preparation

Ensure your dataset is in standard YOLO format with a `data.yaml` file:

```yaml
# data.yaml example
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 12
names: ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook', 'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']
```

**Directory Structure:**
```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│       ├── image001.jpg
│       ├── image002.jpg
│       └── ...
└── labels/
    ├── train/
    ├── val/
    └── test/
        ├── image001.txt
        ├── image002.txt
        └── ...
```

**Label Format (YOLO):**
Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized to `[0, 1]` relative to image dimensions.

### Running Evaluation

```bash
./yolov8_evaluation_runner <data_yaml> <model_path> [image_size] [split]
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `data_yaml` | Yes | - | Path to the dataset configuration file |
| `model_path` | Yes | - | Path to the DRP-AI compiled model directory |
| `image_size` | No | 640 | Input image size for the model |
| `split` | No | test | Dataset split: `train`, `val`, or `test` |

If you want to change the `split`, please also add the `image_size` argument even if you want to keep it as default.

**Example:**

```bash
./yolov8_evaluation_runner ~/data.yaml ~/yolov8_compiled_model/ 640 test
```

### Output Metrics

The evaluator reports the following metrics:

| Metric | Description |
|--------|-------------|
| **P@0.5conf** | Precision at confidence threshold 0.5 |
| **R@0.5conf** | Recall at confidence threshold 0.5 |
| **BestF1** | Best F1 score with optimal confidence threshold |
| **mAP50** | Mean Average Precision at IoU threshold 0.5 |
| **mAP50-95** | Mean Average Precision averaged over IoU thresholds 0.5 to 0.95 |

### Sample Output

```text
=== YOLOv8 Model Evaluation ===
Dataset split:   test
Images dir:      /home/ubuntu/datasets/chess/images/test
Labels dir:      /home/ubuntu/datasets/chess/labels/test
Model path:      /home/ubuntu/models/yolov8n_chess_drpai
Image size:      640
Classes (12): black-bishop, black-king, black-knight...
===============================
[ModelEvaluator] [info] Starting evaluation...
[ModelEvaluator] [info] Loading images from: /home/ubuntu/datasets/chess/images/test
[ModelEvaluator] [info] Loaded 50 images with annotations, skipped 0 images
[ModelEvaluator] [info] Successfully loaded YOLOv8 model.
[ModelEvaluator] [info] Running inference on 50 images...
[ModelEvaluator] [info] Progress: 100.0% (50/50) - Elapsed: 8s
[ModelEvaluator] [info] Inference completed.
[ModelEvaluator] [info] Calculating metrics...
[ModelEvaluator] [info] Calculating AP for: black-bishop
[ModelEvaluator] [info] black-bishop - P@0.5conf: 0.9146, R@0.5conf: 1.0000, BestF1: 0.9900 (conf=0.855), mAP50: 0.9937, mAP50-95: 0.8333
[ModelEvaluator] [info] Calculating AP for: black-king
[ModelEvaluator] [info] black-king - P@0.5conf: 1.0000, R@0.5conf: 1.0000, BestF1: 1.0000 (conf=0.930), mAP50: 1.0000, mAP50-95: 0.9076
...
[ModelEvaluator] [info] ========================================
[ModelEvaluator] [info] Overall Results:
[ModelEvaluator] [info]   Precision@0.5conf: 0.9796
[ModelEvaluator] [info]   Recall@0.5conf:    0.9990
[ModelEvaluator] [info]   Best F1:           0.9971
[ModelEvaluator] [info]   mAP50:             0.9984
[ModelEvaluator] [info]   mAP50-95:          0.8659
[ModelEvaluator] [info] ========================================
[ModelEvaluator] [info] Evaluation completed!
```

## Adding New Models

To add support for a new object detection model architecture (e.g., YOLOv5, YOLOX, MobileNet-SSD):

### 1. Inherit from `ModelEvaluator`

Create a new class that inherits from `model_eval::ModelEvaluator`:

```cpp
// include/rzv_model_evaluation/new_model_evaluation.hpp
#pragma once

#include "rzv_model_evaluation/object_detection_model_evaluation.hpp"

namespace model_eval
{

class NewModelEvaluator : public ModelEvaluator
{
public:
  NewModelEvaluator() = default;
  ~NewModelEvaluator() override = default;

  bool load_model(
    const std::string & model_path,
    const std::vector<std::string> & class_names) override;

  InferenceResult run_inference(const std::string & image_path) override;

private:
  // Your model-specific members
  std::unique_ptr<YourModelWrapper> model_;
};

}  // namespace model_eval
```

### 2. Implement `load_model`

Initialize your specific DRP-AI model wrapper:

```cpp
bool NewModelEvaluator::load_model(
  const std::string & model_path,
  const std::vector<std::string> & class_names)
{
  try {
    model_ = std::make_unique<YourModelWrapper>();
    model_->load(model_path);
    model_->set_class_names(class_names);
    // Configure model parameters...
    return true;
  } catch (const std::exception & e) {
    logger_->error("Failed to load model: {}", e.what());
    return false;
  }
}
```

### 3. Implement `run_inference`

Process an image and return detections:

```cpp
InferenceResult NewModelEvaluator::run_inference(const std::string & image_path)
{
  InferenceResult result(image_path);

  cv::Mat image = load_image(image_path);
  if (image.empty()) {
    return result;
  }

  // Run your model inference
  auto detections = model_->detect(image);

  // Convert to normalized format
  float img_w = static_cast<float>(image.cols);
  float img_h = static_cast<float>(image.rows);

  for (const auto & det : detections) {
    // IMPORTANT: Normalize to [0, 1] in (x_center, y_center, width, height) format
    float x_center = (det.x + det.width / 2.0f) / img_w;
    float y_center = (det.y + det.height / 2.0f) / img_h;
    float width = det.width / img_w;
    float height = det.height / img_h;

    BoundingBox bbox(x_center, y_center, width, height, det.class_id, det.confidence);
    result.detections.emplace_back(bbox, det.confidence, det.class_id);
  }

  return result;
}
```

### 4. Create a Runner Executable

```cpp
// src/new_model_evaluation_runner.cpp
#include "rzv_model_evaluation/new_model_evaluation.hpp"

int main(int argc, char * argv[])
{
  // Parse arguments and run evaluation
  model_eval::NewModelEvaluator evaluator;
  evaluator.set_model_path(model_path);
  evaluator.set_class_names(class_names);
  evaluator.evaluate(images_dir, labels_dir);
  return 0;
}
```

### 5. Update CMakeLists.txt

```cmake
# Add library
add_library(new_model_evaluation src/new_model_evaluation.cpp)
target_link_libraries(new_model_evaluation
  object_detection_model_evaluation
  your_model_library
)

# Add executable
add_executable(new_model_evaluation_runner src/new_model_evaluation_runner.cpp)
target_link_libraries(new_model_evaluation_runner new_model_evaluation)

install(TARGETS
  new_model_evaluation
  new_model_evaluation_runner
  DESTINATION lib/${PROJECT_NAME}
)
```

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for details.

## Copyright

Copyright (C) 2026 Renesas Electronics Corporation and/or its licensors.