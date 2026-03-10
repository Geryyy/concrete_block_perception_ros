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
  TeaserRegistrationParams teaser;

  bool publish_debug_cutout{true};
  bool publish_debug_mask{true};
  bool verbose_logs{true};
  bool dump_enabled{false};
  bool dump_failure_package{true};
  std::string dump_dir;
  std::string fk_seed_tcp_frame{"elastic/K8_tool_center_point"};
  Eigen::Vector3d fk_seed_tcp_to_block_xyz{Eigen::Vector3d::Zero()};
};

BlockRegistrationConfig
load_registration_config(rclcpp::Node & node);

}
