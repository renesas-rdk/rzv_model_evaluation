cmake_minimum_required(VERSION 3.8)
project(rzv_model_evaluation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find all packages
find_package(ament_cmake REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(rzv_model REQUIRED)
find_package(rzv_yolov8 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# create base library
set(TARGET_BASE_LIB rzv_model_evaluation)
add_library(${TARGET_BASE_LIB} SHARED
  src/object_detection_model_evaluation.cpp
)

target_include_directories(${TARGET_BASE_LIB}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${TARGET_BASE_LIB}
  rzv_model
)

target_link_libraries(${TARGET_BASE_LIB}
  ${OpenCV_LIBRARIES}
  spdlog
  fmt
)

# Create library to evaluate the yolov8 detect model
set(TARGET_YOLOV8_LIB yolov8_detect_model_evaluation)
add_library(${TARGET_YOLOV8_LIB} SHARED
  src/yolov8_detect_model_evaluation.cpp
)

target_include_directories(${TARGET_YOLOV8_LIB}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${TARGET_YOLOV8_LIB}
  rzv_yolov8
)

target_link_libraries(${TARGET_YOLOV8_LIB}
  ${TARGET_BASE_LIB}
  ${OpenCV_LIBRARIES}
  spdlog
  fmt
)

# create executable for running the yolov8 evaluate
set(TARGET_YOLOV8_EVAL_RUNNER yolov8_evaluation_runner)
add_executable(${TARGET_YOLOV8_EVAL_RUNNER}
  src/yolov8_evaluation_runner.cpp
)

target_include_directories(${TARGET_YOLOV8_EVAL_RUNNER}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

ament_target_dependencies(${TARGET_YOLOV8_EVAL_RUNNER}
  rzv_yolov8
)

target_link_libraries(${TARGET_YOLOV8_EVAL_RUNNER}
  ${TARGET_YOLOV8_LIB}
  ${OpenCV_LIBRARIES}
  spdlog
  fmt
  yaml-cpp
)

# Install headers
install(
  DIRECTORY include/
  DESTINATION include
)

# Install libraries
install(
  TARGETS ${TARGET_BASE_LIB} ${TARGET_YOLOV8_LIB}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install executable
install(
  TARGETS ${TARGET_YOLOV8_EVAL_RUNNER}
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()