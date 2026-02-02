//***********************************************************************************************************************
// Copyright (C) 2026 Renesas Electronics Corporation and/or its licensors.
// SPDX-License-Identifier: AGPL-3.0
//***********************************************************************************************************************
#include "rzv_model_evaluation/object_detection_model_evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace model_eval
{

ModelEvaluator::ModelEvaluator()
{
  // Try to retrieve an existing logger named "ModelEvaluator"
  logger_ = spdlog::get("ModelEvaluator");
  if (!logger_) {
    // The logger was not registered, so need do it.
    logger_ = spdlog::stdout_color_mt("ModelEvaluator");

    // Set the log level to info by default
    logger_->set_level(spdlog::level::info);
  }

  // Load levels from environment if set
  spdlog::cfg::load_env_levels();

  // Set log message format
  logger_->set_pattern("[ModelEvaluator] [%^%l%$] %v");
}

// Implementation of structures
BoundingBox::BoundingBox(float x, float y, float w, float h, int cls, float conf)
: x(x), y(y), width(w), height(h), class_id(cls), confidence(conf)
{
}

Detection::Detection(const BoundingBox & box, float conf, int cls)
: bbox(box), confidence(conf), class_id(cls)
{
}

GroundTruth::GroundTruth(const std::string & path) : image_path(path) {}

InferenceResult::InferenceResult(const std::string & path) : image_path(path) {}

// Implementation of ModelEvaluator methods

const std::vector<GroundTruth> & ModelEvaluator::get_ground_truths() const
{
  return ground_truths_;
}

const std::vector<InferenceResult> & ModelEvaluator::get_inference_results() const
{
  return inference_results_;
}

void ModelEvaluator::set_class_names(const std::vector<std::string> & class_names)
{
  class_names_ = class_names;
}

void ModelEvaluator::set_model_path(const std::string & model_path) { model_path_ = model_path; }

bool ModelEvaluator::is_image_file(const std::string & filename) const
{
  std::string extension = std::filesystem::path(filename).extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
         extension == ".bmp" || extension == ".tiff" || extension == ".tif";
}

std::string ModelEvaluator::get_base_name(const std::string & filepath) const
{
  return std::filesystem::path(filepath).stem().string();
}

cv::Mat ModelEvaluator::load_image(const std::string & image_path)
{
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Warning: Could not load image: " << image_path << std::endl;
  }
  return image;
}

// Load dataset (images + annotations) from specified directories
bool ModelEvaluator::load_data(const std::string & images_dir, const std::string & annotations_dir)
{
  ground_truths_.clear();

  // Quick existence check
  if (!std::filesystem::exists(images_dir)) {
    logger_->error("Images directory does not exist: {}", images_dir);
    return false;
  }

  if (!std::filesystem::exists(annotations_dir)) {
    logger_->error("Annotations directory does not exist: {}", annotations_dir);
    return false;
  }

  logger_->info("Loading images from: {}", images_dir);
  logger_->info("Loading annotations from: {}", annotations_dir);

  int processed_images = 0;
  int skipped_images = 0;

  try {
    for (const auto & entry : std::filesystem::directory_iterator(images_dir)) {
      if (entry.is_regular_file()) {
        std::string image_path = entry.path().string();

        // Only process image files
        if (is_image_file(image_path)) {
          std::string base_name = get_base_name(image_path);
          std::string txt_path = annotations_dir + "/" + base_name + ".txt";

          // Check if corresponding annotation file exists
          if (std::filesystem::exists(txt_path)) {
            GroundTruth gt(image_path);
            if (load_annotation_file(txt_path, gt)) {
              ground_truths_.push_back(gt);
              processed_images++;
            } else {
              logger_->error("Failed to load annotations for image '{}', skipping...", base_name);
              skipped_images++;
            }
          } else {
            logger_->warn("Annotation file not found for image '{}', skipping...", base_name);
            skipped_images++;
          }
        }
      }
    }
  } catch (const std::exception & e) {
    logger_->error("Exception while loading data: {}", e.what());
    return false;
  }

  logger_->info(
    "Loaded {} images with annotations, skipped {} images without valid annotations",
    processed_images, skipped_images);

  if (ground_truths_.empty()) {
    logger_->error("No valid images with annotations were loaded.");
    return false;
  }

  return true;
}

// Load annotation file from a text file
// Expect format per line: class_id x_center y_center width height (normalized)
bool ModelEvaluator::load_annotation_file(const std::string & txt_path, GroundTruth & gt)
{
  std::ifstream file(txt_path);
  if (!file.is_open()) {
    logger_->error("Could not open annotation file: {}", txt_path);
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    std::istringstream iss(line);
    int class_id;
    float x, y, width, height;

    // Expect the annotations in format: class_id x y width height
    if (iss >> class_id >> x >> y >> width >> height) {
      BoundingBox bbox(x, y, width, height, class_id);
      gt.bboxes.push_back(bbox);
    } else {
      logger_->warn("Invalid annotation line in file '{}': {}", txt_path, line);
    }
  }

  file.close();
  return true;
}

std::map<std::string, float> ModelEvaluator::evaluate(
  const std::string & images_dir, const std::string & annotations_dir)
{
  // Check if model path as well as class names are set
  if (model_path_.empty()) {
    logger_->error("Model path is not set. Cannot evaluate.");
    return {};
  }

  if (class_names_.empty()) {
    logger_->error("Class names are not set. Cannot evaluate.");
    return {};
  }

  std::map<std::string, float> results;

  logger_->info("Starting evaluation...");

  // Step 1: Load images and annotations
  if (!load_data(images_dir, annotations_dir)) {
    logger_->error("Failed to load data from directories.");
    return results;
  }

  // Step 2: Load model
  logger_->info("Loading model from: {}", model_path_);
  if (!load_model(model_path_, class_names_)) {
    logger_->error("Failed to load model from: {}", model_path_);
    return results;
  }

  // Step 3: Run inference on all images
  logger_->info("Running inference on {} images...", ground_truths_.size());
  inference_results_.clear();

  const size_t total = ground_truths_.size();
  const size_t log_interval = std::max(total / 10, size_t(1));

  auto start_time = std::chrono::steady_clock::now();

  for (size_t i = 0; i < total; ++i) {
    const auto & gt = ground_truths_[i];

    InferenceResult result = run_inference(gt.image_path);
    result.image_path = gt.image_path;
    inference_results_.push_back(result);

    // Log progress
    if ((i + 1) % log_interval == 0 || (i + 1) == total) {
      float progress = static_cast<float>(i + 1) / total * 100.0f;

      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - start_time)
                       .count();

      logger_->info("Progress: {:.1f}% ({}/{}) - Elapsed: {}s", progress, i + 1, total, elapsed);
    }
  }

  logger_->info("Inference completed.");

  // Step 4: Calculate mAP50-95
  logger_->info("Calculating mAP50-95...");
  calculate_map_50_95();

  logger_->info("Evaluation completed!");
  return results;
}

float ModelEvaluator::calculate_iou(const BoundingBox & box1, const BoundingBox & box2)
{
  // Convert from center format to corner format
  float box1_x1 = box1.x - box1.width / 2.0f;
  float box1_y1 = box1.y - box1.height / 2.0f;
  float box1_x2 = box1.x + box1.width / 2.0f;
  float box1_y2 = box1.y + box1.height / 2.0f;

  float box2_x1 = box2.x - box2.width / 2.0f;
  float box2_y1 = box2.y - box2.height / 2.0f;
  float box2_x2 = box2.x + box2.width / 2.0f;
  float box2_y2 = box2.y + box2.height / 2.0f;

  // Calculate intersection
  float inter_x1 = std::max(box1_x1, box2_x1);
  float inter_y1 = std::max(box1_y1, box2_y1);
  float inter_x2 = std::min(box1_x2, box2_x2);
  float inter_y2 = std::min(box1_y2, box2_y2);

  // Check for non-overlapping boxes
  float inter_w = std::max(0.0f, inter_x2 - inter_x1);
  float inter_h = std::max(0.0f, inter_y2 - inter_y1);

  float intersection = inter_w * inter_h;
  float area1 = box1.width * box1.height;
  float area2 = box2.width * box2.height;
  float union_area = area1 + area2 - intersection;

  // Add a small epsilon to avoid division by zero
  float iou = (union_area > 1e-6f) ? intersection / union_area : 0.0f;

  return iou;
}

std::set<int> ModelEvaluator::get_all_class_ids() const
{
  std::set<int> class_ids;

  // Collect class IDs from ground truth
  for (const auto & gt : ground_truths_) {
    for (const auto & bbox : gt.bboxes) {
      class_ids.insert(bbox.class_id);
    }
  }

  // Collect class IDs from inference results
  for (const auto & result : inference_results_) {
    for (const auto & detection : result.detections) {
      class_ids.insert(detection.class_id);
    }
  }

  return class_ids;
}

void ModelEvaluator::calculate_map_50_95()
{
  std::set<int> all_classes = get_all_class_ids();

  if (all_classes.empty()) {
    logger_->warn("No classes found in ground truths or detections.");
    return;
  }

  float total_map_50_95 = 0.0f;
  float total_map_50 = 0.0f;
  float total_precision = 0.0f;
  float total_recall = 0.0f;
  float total_best_f1 = 0.0f;

  for (int class_id : all_classes) {
    logger_->info("Calculating AP for: {}", class_names_.at(class_id));

    // Calculate AP at IoU thresholds from 0.5 to 0.95
    float class_map_50_95 = 0.0f;
    float map_50 = 0.0f;

    for (int iou_step = 0; iou_step < 10; ++iou_step) {
      float iou_thresh = 0.5f + iou_step * 0.05f;
      float ap = calculate_ap_for_class(class_id, iou_thresh);
      class_map_50_95 += ap;
      if (iou_step == 0) map_50 = ap;
    }
    class_map_50_95 /= 10.0f;

    // Calculate Precision and Recall at confidence threshold 0.5
    float precision = 0.0f;
    float recall = 0.0f;
    calculate_precision_recall_for_class(class_id, 0.5f, precision, recall, 0.5f);

    // Calculate best F1 operating point
    float best_precision = 0.0f;
    float best_recall = 0.0f;
    float best_f1 = 0.0f;
    float best_conf = 0.0f;
    calculate_best_precision_recall_for_class(
      class_id, 0.5f, best_precision, best_recall, best_f1, best_conf);

    // Accumulate totals
    total_map_50_95 += class_map_50_95;
    total_map_50 += map_50;
    total_precision += precision;
    total_recall += recall;
    total_best_f1 += best_f1;

    logger_->info(
      "{} - P@0.5conf: {:.4f}, R@0.5conf: {:.4f}, BestF1: {:.4f} (conf={:.3f}), mAP50: {:.4f}, "
      "mAP50-95: {:.4f}",
      class_names_.at(class_id), precision, recall, best_f1, best_conf, map_50, class_map_50_95);
  }

  // Final overall results
  float num_classes = static_cast<float>(all_classes.size());

  logger_->info("========================================");
  logger_->info("Overall Results:");
  logger_->info("  Precision@0.5conf: {:.4f}", total_precision / num_classes);
  logger_->info("  Recall@0.5conf:    {:.4f}", total_recall / num_classes);
  logger_->info("  Best F1:           {:.4f}", total_best_f1 / num_classes);
  logger_->info("  mAP50:             {:.4f}", total_map_50 / num_classes);
  logger_->info("  mAP50-95:          {:.4f}", total_map_50_95 / num_classes);
  logger_->info("========================================");
}

float ModelEvaluator::calculate_ap_for_class(int class_id, float iou_threshold)
{
  // Gather all detections and ground truths for the specified class
  std::vector<DetectionWithImage> all_detections;
  std::vector<GTWithImage> all_gt_boxes;

  for (size_t i = 0; i < inference_results_.size(); ++i) {
    int det_count = 0;
    int gt_count = 0;

    for (const auto & detection : inference_results_[i].detections) {
      if (detection.class_id == class_id) {
        all_detections.emplace_back(detection, i);
        det_count++;
      }
    }

    for (const auto & bbox : ground_truths_[i].bboxes) {
      if (bbox.class_id == class_id) {
        all_gt_boxes.emplace_back(bbox, i);
        gt_count++;
      }
    }
  }

  if (all_gt_boxes.empty()) {
    return 0.0f;
  }

  // Sort detections by confidence (descending)
  std::sort(
    all_detections.begin(), all_detections.end(),
    [](const DetectionWithImage & a, const DetectionWithImage & b) {
      return a.detection.confidence > b.detection.confidence;
    });

  // Track which GT boxes have been matched
  std::vector<bool> gt_matched(all_gt_boxes.size(), false);

  // Build PR curve
  std::vector<std::pair<float, float>> pr_curve;
  int tp = 0;
  int fp = 0;

  for (const auto & det_with_img : all_detections) {
    float best_iou = 0.0f;
    int best_gt_idx = -1;

    // Find the best matching GT box for this detection
    for (size_t j = 0; j < all_gt_boxes.size(); ++j) {
      if (gt_matched[j]) continue;
      if (all_gt_boxes[j].image_idx != det_with_img.image_idx) continue;

      float iou = calculate_iou(det_with_img.detection.bbox, all_gt_boxes[j].bbox);

      if (iou > best_iou && iou >= iou_threshold) {
        best_iou = iou;
        best_gt_idx = static_cast<int>(j);
      }
    }

    if (best_gt_idx >= 0) {
      gt_matched[best_gt_idx] = true;
      tp++;
    } else {
      fp++;
    }

    float precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
    float recall = static_cast<float>(tp) / all_gt_boxes.size();
    pr_curve.push_back({recall, precision});
  }

  float ap = calculate_ap_from_pr_curve(pr_curve);

  logger_->debug("Class {} IoU={:.2f}: AP={:.4f}", class_id, iou_threshold, ap);

  return ap;
}

float ModelEvaluator::calculate_ap_from_pr_curve(
  const std::vector<std::pair<float, float>> & pr_curve)
{
  if (pr_curve.empty()) {
    return 0.0f;
  }

  // Make a copy to modify
  std::vector<std::pair<float, float>> curve = pr_curve;

  // Make precision monotonically decreasing
  for (int i = static_cast<int>(curve.size()) - 2; i >= 0; --i) {
    curve[i].second = std::max(curve[i].second, curve[i + 1].second);
  }

  // Calculate AP using the trapezoidal rule
  float ap = 0.0f;
  float prev_recall = 0.0f;

  for (const auto & point : curve) {
    float recall = point.first;
    float precision = point.second;

    ap += (recall - prev_recall) * precision;
    prev_recall = recall;
  }

  return ap;
}

void ModelEvaluator::calculate_precision_recall_for_class(
  int class_id, float iou_threshold, float & precision, float & recall,
  float confidence_threshold)  // Add confidence threshold parameter
{
  int true_positives = 0;
  int false_positives = 0;
  int total_gt = 0;

  for (size_t i = 0; i < inference_results_.size(); ++i) {
    // Get ground truths for this class in this image
    std::vector<BoundingBox> gt_boxes;
    for (const auto & bbox : ground_truths_[i].bboxes) {
      if (bbox.class_id == class_id) {
        gt_boxes.push_back(bbox);
      }
    }
    total_gt += gt_boxes.size();

    // Get detections for this class in this image
    // FILTER BY CONFIDENCE THRESHOLD
    std::vector<Detection> detections;
    for (const auto & det : inference_results_[i].detections) {
      if (det.class_id == class_id && det.confidence >= confidence_threshold) {
        detections.push_back(det);
      }
    }

    // Sort detections by confidence (descending)
    std::sort(detections.begin(), detections.end(), [](const Detection & a, const Detection & b) {
      return a.confidence > b.confidence;
    });

    // Track which ground truths have been matched
    std::vector<bool> gt_matched(gt_boxes.size(), false);

    // Match detections to ground truths
    for (const auto & det : detections) {
      float best_iou = 0.0f;
      int best_gt_idx = -1;

      for (size_t j = 0; j < gt_boxes.size(); ++j) {
        if (gt_matched[j]) continue;

        float iou = calculate_iou(det.bbox, gt_boxes[j]);
        if (iou > best_iou && iou >= iou_threshold) {
          best_iou = iou;
          best_gt_idx = static_cast<int>(j);
        }
      }

      if (best_gt_idx >= 0) {
        true_positives++;
        gt_matched[best_gt_idx] = true;
      } else {
        false_positives++;
      }
    }
  }

  precision = (true_positives + false_positives > 0)
                ? static_cast<float>(true_positives) / (true_positives + false_positives)
                : 0.0f;

  recall = (total_gt > 0) ? static_cast<float>(true_positives) / total_gt : 0.0f;
}

void ModelEvaluator::calculate_best_precision_recall_for_class(
  int class_id, float iou_threshold, float & best_precision, float & best_recall, float & best_f1,
  float & best_conf_threshold)
{
  // Gather all detections and ground truths for the specified class
  std::vector<DetectionWithImage> all_detections;
  std::vector<GTWithImage> all_gt_boxes;

  for (size_t i = 0; i < inference_results_.size(); ++i) {
    for (const auto & detection : inference_results_[i].detections) {
      if (detection.class_id == class_id) {
        all_detections.emplace_back(detection, i);
      }
    }

    for (const auto & bbox : ground_truths_[i].bboxes) {
      if (bbox.class_id == class_id) {
        all_gt_boxes.emplace_back(bbox, i);
      }
    }
  }

  if (all_gt_boxes.empty() || all_detections.empty()) {
    best_precision = 0.0f;
    best_recall = 0.0f;
    best_f1 = 0.0f;
    best_conf_threshold = 0.0f;
    return;
  }

  // Sort detections by confidence (descending)
  std::sort(
    all_detections.begin(), all_detections.end(),
    [](const DetectionWithImage & a, const DetectionWithImage & b) {
      return a.detection.confidence > b.detection.confidence;
    });

  // Track which GT boxes have been matched
  std::vector<bool> gt_matched(all_gt_boxes.size(), false);

  int tp = 0;
  int fp = 0;
  int total_gt = static_cast<int>(all_gt_boxes.size());

  best_f1 = 0.0f;
  best_precision = 0.0f;
  best_recall = 0.0f;
  best_conf_threshold = 0.0f;

  for (const auto & det_with_img : all_detections) {
    float best_iou = 0.0f;
    int best_gt_idx = -1;

    // Find the best matching GT box for this detection
    for (size_t j = 0; j < all_gt_boxes.size(); ++j) {
      if (gt_matched[j]) continue;
      if (all_gt_boxes[j].image_idx != det_with_img.image_idx) continue;

      float iou = calculate_iou(det_with_img.detection.bbox, all_gt_boxes[j].bbox);

      if (iou > best_iou && iou >= iou_threshold) {
        best_iou = iou;
        best_gt_idx = static_cast<int>(j);
      }
    }

    if (best_gt_idx >= 0) {
      gt_matched[best_gt_idx] = true;
      tp++;
    } else {
      fp++;
    }

    // Calculate precision, recall, and F1 at this point
    float precision = static_cast<float>(tp) / (tp + fp);
    float recall = static_cast<float>(tp) / total_gt;
    float f1 = (precision + recall > 0) ? 2.0f * precision * recall / (precision + recall) : 0.0f;

    // Track best F1 score
    if (f1 > best_f1) {
      best_f1 = f1;
      best_precision = precision;
      best_recall = recall;
      best_conf_threshold = det_with_img.detection.confidence;
    }
  }
}

}  // namespace model_eval