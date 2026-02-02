//***********************************************************************************************************************
// Copyright (C) 2026 Renesas Electronics Corporation and/or its licensors.
// SPDX-License-Identifier: AGPL-3.0
//***********************************************************************************************************************
#pragma once
#include "rzv_model_evaluation/object_detection_model_evaluation.hpp"
#include "rzv_yolov8/yolov8_detect_model.hpp"

namespace model_eval
{
class YOLOv8DetectModelEvaluator : public ModelEvaluator
{
public:
  YOLOv8DetectModelEvaluator();
  virtual ~YOLOv8DetectModelEvaluator() = default;
  virtual bool load_model(const std::string & model_type, const std::vector<std::string> & class_names) override;
  virtual InferenceResult run_inference(const std::string & image_path) override;

  // For YOLOv8 model, we have to export the set image size method
  void set_image_size(int size);

private:
  std::unique_ptr<rzv_model::YOLOv8DetectModel> model_;
};
}  // namespace model_eval