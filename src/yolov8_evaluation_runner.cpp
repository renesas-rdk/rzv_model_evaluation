//***********************************************************************************************************************
// Copyright (C) 2026 Renesas Electronics Corporation and/or its licensors.
// SPDX-License-Identifier: AGPL-3.0
//***********************************************************************************************************************
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "rzv_model_evaluation/yolov8_detect_model_evaluation.hpp"

void print_usage(const char * program_name)
{
  std::cerr << "Usage: " << program_name << " <data_yaml> <model_path> [image_size] [split]\n"
            << "\nArguments:\n"
            << "  data_yaml    Path to data.yaml file (YOLO/Roboflow format)\n"
            << "  model_path   Path to DRP-AI compiled model file\n"
            << "  image_size   (Optional) Input image size, default: 640\n"
            << "  split        (Optional) Dataset split: train/val/test, default: test\n"
            << "\nExample:\n"
            << "  " << program_name << " ./data.yaml ./yolov8_model 640 test\n"
            << std::endl;
}

struct DatasetConfig
{
  std::string images_dir;
  std::string labels_dir;
  std::vector<std::string> class_names;
  int num_classes = 0;
};

DatasetConfig load_dataset_config(const std::string & yaml_path, const std::string & split = "test")
{
  DatasetConfig config;

  try {
    if (!std::filesystem::exists(yaml_path)) {
      std::cerr << "Error: YAML file not found: " << yaml_path << std::endl;
      return config;
    }

    YAML::Node root = YAML::LoadFile(yaml_path);
    std::string base_dir = std::filesystem::path(yaml_path).parent_path().string();

    // Get images directory for the specified split
    if (!root[split]) {
      std::cerr << "Error: Split '" << split << "' not found in YAML" << std::endl;
      return config;
    }

    std::string images_path = root[split].as<std::string>();

    // Handle relative paths
    if (images_path[0] == '.') {
      config.images_dir = base_dir + "/" + images_path;
    } else {
      config.images_dir = images_path;
    }

    // Labels directory is typically parallel to images (images -> labels)
    config.labels_dir = config.images_dir;
    size_t pos = config.labels_dir.rfind("/images");
    if (pos != std::string::npos) {
      config.labels_dir.replace(pos, 7, "/labels");
    }

    // Get number of classes
    if (root["nc"]) {
      config.num_classes = root["nc"].as<int>();
    }

    // Get class names
    if (root["names"]) {
      for (const auto & name : root["names"]) {
        config.class_names.push_back(name.as<std::string>());
      }
    }

    // Validate
    if (config.class_names.size() != static_cast<size_t>(config.num_classes)) {
      std::cerr << "Warning: nc=" << config.num_classes << " but found "
                << config.class_names.size() << " class names" << std::endl;
    }

  } catch (const std::exception & e) {
    std::cerr << "Error parsing YAML: " << e.what() << std::endl;
  }

  return config;
}

int main(int argc, char ** argv)
{
  if (argc < 3 || argc > 5) {
    print_usage(argv[0]);
    return 1;
  }

  std::string data_yaml = argv[1];
  std::string model_path = argv[2];
  int image_size = (argc >= 4) ? std::stoi(argv[3]) : 640;
  std::string split = (argc >= 5) ? argv[4] : "test";

  // Load dataset configuration from YAML
  auto config = load_dataset_config(data_yaml, split);

  if (config.class_names.empty()) {
    std::cerr << "Error: Failed to load dataset configuration" << std::endl;
    return 1;
  }

  if (!std::filesystem::exists(config.images_dir)) {
    std::cerr << "Error: Images directory not found: " << config.images_dir << std::endl;
    return 1;
  }

  if (!std::filesystem::exists(config.labels_dir)) {
    std::cerr << "Error: Labels directory not found: " << config.labels_dir << std::endl;
    return 1;
  }

  std::cout << "=== YOLOv8 Model Evaluation ===" << std::endl;
  std::cout << "Dataset split:   " << split << std::endl;
  std::cout << "Images dir:      " << config.images_dir << std::endl;
  std::cout << "Labels dir:      " << config.labels_dir << std::endl;
  std::cout << "Model path:      " << model_path << std::endl;
  std::cout << "Image size:      " << image_size << std::endl;
  std::cout << "Classes (" << config.class_names.size() << "): ";
  for (size_t i = 0; i < config.class_names.size(); ++i) {
    std::cout << config.class_names[i];
    if (i < config.class_names.size() - 1) std::cout << ", ";
  }
  std::cout << "\n===============================" << std::endl;

  // Create evaluator and run
  auto runner = std::make_unique<model_eval::YOLOv8DetectModelEvaluator>();
  // For YOLOv8, we need to set image size, other models may not need this
  runner->set_image_size(image_size);
  runner->set_model_path(model_path);
  runner->set_class_names(config.class_names);

  auto results = runner->evaluate(config.images_dir, config.labels_dir);

  return 0;
}