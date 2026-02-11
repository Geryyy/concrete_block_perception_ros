#pragma once

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

#include "pcd_block_estimation/template_utils.hpp"
#include "concrete_block_perception/registration/block_registration_pipeline.hpp"

namespace concrete_block_perception
{

struct BlockRegistrationConfig
{
  std::string world_frame;

  Eigen::Matrix4d T_P_C;
  Eigen::Matrix3d K;
  std::vector<pcd_block::TemplateData> templates;

  PreprocessingParams preproc;
  GlobalRegistrationParams glob;
  LocalRegistrationParams local;

  bool publish_debug_cutout{true};
  bool publish_debug_mask{true};
  bool dump_enabled{false};
  std::string dump_dir;
};

BlockRegistrationConfig
load_registration_config(rclcpp::Node & node);

}
