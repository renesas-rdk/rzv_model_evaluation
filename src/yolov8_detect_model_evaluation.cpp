//***********************************************************************************************************************
// Copyright (C) 2026 Renesas Electronics Corporation and/or its licensors.
// SPDX-License-Identifier: AGPL-3.0
//***********************************************************************************************************************
#include "rzv_model_evaluation/yolov8_detect_model_evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

namespace model_eval
{
YOLOv8DetectModelEvaluator::YOLOv8DetectModelEvaluator()
: ModelEvaluator(), model_(std::make_unique<rzv_model::YOLOv8DetectModel>())
{
  logger_->info("YOLOv8DetectModelEvaluator initialized.");
}

void YOLOv8DetectModelEvaluator::set_image_size(int size) { model_->set_image_size(size); }

bool YOLOv8DetectModelEvaluator::load_model(
  const std::string & model_path, const std::vector<std::string> & class_names)
{
  // Configure model
  model_->set_class_names(class_names);
  model_->set_confidence_threshold(0.001f);  // Low threshold to capture all detections
  model_->set_nms_threshold(0.64f);          // COCO recommended NMS threshold
  model_->set_dfl_sigmoid_mode(rzv_model::DFLSigmoidMode::InDfl);

  if (!model_->load(model_path)) {
    logger_->error("Failed to load YOLOv8 model from: {}", model_path);
    return false;
  }
  logger_->info("Successfully loaded YOLOv8 model.");
  return true;
}

InferenceResult YOLOv8DetectModelEvaluator::run_inference(const std::string & image_path)
{
  logger_->debug("Running inference on image: {}", image_path);
  InferenceResult result(image_path);

  cv::Mat image = load_image(image_path);

  // Get image dimensions for normalization
  float image_width = static_cast<float>(image.cols);
  float image_height = static_cast<float>(image.rows);

  cv::Mat yuv422_image = rzv_model::Utils::bgr_to_yuv422(image, rzv_model::YUV422Format::YUYV);

  auto object_image_input =
    rzv_model::ModelInput{yuv422_image, cv::Rect(0, 0, yuv422_image.cols, yuv422_image.rows)};
  auto inference_result = model_->run<rzv_model::YOLOv8DetectionResult>(object_image_input);

  // Convert rzv_model::YOLOv8DetectionResult to model_eval::Detection
  for (const auto & detection : inference_result->detections) {
    if (detection.is_valid) {
      // The resulting bbox is in (x, y, width, height) format where (x, y) is the top-left corner
      // Convert it to YOLO format: x_center, y_center, width, height (normalized 0-1)

      float x_center_normalized = (detection.bbox.x + detection.bbox.width / 2.0f) / image_width;
      float y_center_normalized = (detection.bbox.y + detection.bbox.height / 2.0f) / image_height;
      float width_normalized = detection.bbox.width / image_width;
      float height_normalized = detection.bbox.height / image_height;

      BoundingBox bbox(
        x_center_normalized, y_center_normalized, width_normalized, height_normalized,
        detection.class_id, detection.confidence);

      result.detections.emplace_back(bbox, detection.confidence, detection.class_id);

      logger_->debug(
        "Detection: class={}, conf={:.3f}, bbox(normalized)=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]",
        detection.class_id, detection.confidence, x_center_normalized, y_center_normalized,
        width_normalized, height_normalized);
    }
  }

  logger_->debug(
    "Inference completed for image: {} with {} detections", image_path, result.detections.size());
  return result;
}

}  // namespace model_eval