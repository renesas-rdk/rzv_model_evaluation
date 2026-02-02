//***********************************************************************************************************************
// Copyright (C) 2026 Renesas Electronics Corporation and/or its licensors.
// SPDX-License-Identifier: AGPL-3.0
//***********************************************************************************************************************
#pragma once
#include <map>
#include <opencv2/core.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "rzv_model/utils.hpp"
#include "spdlog/cfg/env.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace model_eval
{

// Data structure to hold bounding box information
// The format is (x, y, width, height) where (x, y) is the center of the box and width, height are the dimensions with normalized values [0,1]
struct BoundingBox
{
  float x, y, width, height;
  int class_id;
  float confidence;

  BoundingBox(float x = 0, float y = 0, float w = 0, float h = 0, int cls = -1, float conf = 0.0f);
};

struct Detection
{
  BoundingBox bbox;
  float confidence;
  int class_id;

  Detection(const BoundingBox & box, float conf, int cls);
};

// Data structure to hold ground truth annotations for an image
struct GroundTruth
{
  std::vector<BoundingBox> bboxes;
  std::string image_path;

  GroundTruth(const std::string & path = "");
};

struct InferenceResult
{
  std::vector<Detection> detections;
  std::string image_path;

  InferenceResult(const std::string & path = "");
};

struct DetectionWithImage
{
  Detection detection;
  size_t image_idx;

  DetectionWithImage(const Detection & det, size_t idx) : detection(det), image_idx(idx) {}
};

struct GTWithImage
{
  BoundingBox bbox;
  size_t image_idx;

  GTWithImage(const BoundingBox & box, size_t idx) : bbox(box), image_idx(idx) {}
};

class ModelEvaluator
{
public:
  ModelEvaluator();
  virtual ~ModelEvaluator() = default;

  // Pure virtual methods to be implemented by derived classes
  virtual bool load_model(
    const std::string & model_type, const std::vector<std::string> & class_names) = 0;
  virtual InferenceResult run_inference(const std::string & image_path) = 0;

  // Helper method to load dataset (images + annotations)
  bool load_data(const std::string & data_dir, const std::string & annotations_dir);

  // Method to evaluate model and compute mAP50-95
  std::map<std::string, float> evaluate(
    const std::string & data_dir, const std::string & annotations_dir);

  // Getter for ground truths
  const std::vector<GroundTruth> & get_ground_truths() const;

  // Getter for inference results
  const std::vector<InferenceResult> & get_inference_results() const;

  // Setters for class names and model path
  void set_class_names(const std::vector<std::string> & class_names);
  void set_model_path(const std::string & model_path);

protected:
  std::vector<GroundTruth> ground_truths_;
  std::vector<InferenceResult> inference_results_;
  std::vector<std::string> class_names_ = {};
  std::string model_path_ = "";
  std::shared_ptr<spdlog::logger> logger_;

  // Helper methods
  bool load_annotation_file(const std::string & txt_path, GroundTruth & gt);
  cv::Mat load_image(const std::string & image_path);
  float calculate_iou(const BoundingBox & box1, const BoundingBox & box2);
  void calculate_map_50_95();
  float calculate_ap_for_class(int class_id, float iou_threshold);
  float calculate_ap_from_pr_curve(const std::vector<std::pair<float, float>> & pr_curve);
  void calculate_precision_recall_for_class(
    int class_id, float iou_threshold, float & precision, float & recall,
    float confidence_threshold = 0.5f);
  void calculate_best_precision_recall_for_class(
    int class_id, float iou_threshold, float & best_precision, float & best_recall, float & best_f1,
    float & best_conf_threshold);
  std::set<int> get_all_class_ids() const;

  // Utility methods
  bool is_image_file(const std::string & filename) const;
  std::string get_base_name(const std::string & filepath) const;
};

}  // namespace model_eval