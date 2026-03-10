#pragma once

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

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
  bool enable_cluster_filter{false};
  double cluster_eps{0.08};
  int cluster_min_points{20};
  int cluster_min_size{100};
};

struct GlobalRegistrationParams
{
  static constexpr int MAX_PLANES = 2;
  Eigen::Vector3d Z_WORLD{0, 0, 1};
  double dist_thresh{0.02};
  int min_inliers{100};
  double angle_thresh{0.9};
  double max_plane_center_dist{0.6};
  bool enable_plane_clipping{false};
  bool reject_tall_vertical{true};
};

struct LocalRegistrationParams
{
  double icp_dist{0.04};
  bool relax_num_faces_match{false};
  bool use_fk_translation_seed{false};
  std::vector<double> icp_dist_multipliers{1.0, 1.5, 2.0};
  bool enable_point_to_point_fallback{true};
};

struct TeaserRegistrationParams
{
  double noise_bound{0.02};
  double cbar2{1.0};
  bool estimate_scaling{false};
  double rotation_gnc_factor{1.4};
  int rotation_max_iterations{100};
  double rotation_cost_threshold{1e-6};
  double max_clique_time_limit_s{0.2};
  size_t min_correspondences{30};
  size_t max_template_points{1000};
  double nn_corr_max_dist{0.08};
  bool enable_icp_refinement{true};
  double icp_refine_dist{0.04};
  double eval_corr_dist{0.04};
};

struct RegistrationInput
{
  open3d::geometry::PointCloud scene;
  cv::Mat mask;
  Eigen::Matrix4d T_world_cloud;
  bool has_translation_seed_world{false};
  Eigen::Vector3d translation_seed_world{Eigen::Vector3d::Zero()};
};

struct RegistrationOutput
{
  bool success{false};
  Eigen::Matrix4d T_world_block{Eigen::Matrix4d::Identity()};
  double fitness{0.0};
  double rmse{0.0};
  int template_index{-1};
  std::string failure_stage;
  std::string failure_reason;

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
    const LocalRegistrationParams & loc,
    const rclcpp::Logger & logger,
    bool verbose_logs);

  RegistrationOutput run(const RegistrationInput & in);

private:
  bool computeCutout(
    const open3d::geometry::PointCloud & scene,
    const cv::Mat & mask,
    open3d::geometry::PointCloud & cutout);

  void preprocess(
    open3d::geometry::PointCloud & cutout,
    const Eigen::Matrix4d & T_world_cloud);

  bool keepDominantCluster(
    open3d::geometry::PointCloud & cutout);

  Eigen::Matrix4d T_P_C_;
  Eigen::Matrix3d K_;
  std::vector<pcd_block::TemplateData> templates_;

  PreprocessingParams pre_;
  GlobalRegistrationParams glob_;
  LocalRegistrationParams loc_;
  rclcpp::Logger logger_;
  bool verbose_logs_{false};
};

} // namespace
