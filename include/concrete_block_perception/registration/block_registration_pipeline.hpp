#pragma once

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>
#include <opencv2/core.hpp>

#include "pcd_block_estimation/pose_estimation.hpp"
#include "pcd_block_estimation/mask_projection.hpp"
#include "pcd_block_estimation/template_utils.hpp"

namespace concrete_block_perception
{

struct PreprocessingParams
{
  size_t max_pts{500};
  int nb_neighbors{20};
  double std_dev{2.0};
};

struct GlobalRegistrationParams
{
  static constexpr int MAX_PLANES = 2;
  Eigen::Vector3d Z_WORLD{0, 0, 1};
  double dist_thresh{0.02};
  int min_inliers{100};
  double angle_thresh{0.9};
  double max_plane_center_dist{0.6};
};

struct LocalRegistrationParams
{
  double icp_dist{0.04};
};

struct RegistrationInput
{
  open3d::geometry::PointCloud scene;
  cv::Mat mask;
  Eigen::Matrix4d T_world_cloud;
};

struct RegistrationOutput
{
  bool success{false};
  Eigen::Matrix4d T_world_block{Eigen::Matrix4d::Identity()};
  double fitness{0.0};
  double rmse{0.0};
  int template_index{-1};

  // optional debug
  open3d::geometry::PointCloud debug_scene;
};

class BlockRegistrationPipeline
{
public:
  BlockRegistrationPipeline(
    const Eigen::Matrix4d & T_P_C,
    const Eigen::Matrix3d & K,
    const std::vector<pcd_block::TemplateData> & templates,
    const PreprocessingParams & pre,
    const GlobalRegistrationParams & glob,
    const LocalRegistrationParams & loc);

  RegistrationOutput run(const RegistrationInput & in);

private:
  bool computeCutout(
    const open3d::geometry::PointCloud & scene,
    const cv::Mat & mask,
    open3d::geometry::PointCloud & cutout);

  void preprocess(
    open3d::geometry::PointCloud & cutout,
    const Eigen::Matrix4d & T_world_cloud);

  Eigen::Matrix4d globalResultToTransform(
    const pcd_block::GlobalRegistrationResult & glob);

  Eigen::Matrix4d T_P_C_;
  Eigen::Matrix3d K_;
  std::vector<pcd_block::TemplateData> templates_;

  PreprocessingParams pre_;
  GlobalRegistrationParams glob_;
  LocalRegistrationParams loc_;
};

} // namespace
