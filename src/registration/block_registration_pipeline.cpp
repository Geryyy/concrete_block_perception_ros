#include "concrete_block_perception/registration/block_registration_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

using namespace open3d;
using namespace pcd_block;

namespace concrete_block_perception
{
namespace
{

double rotationDistanceRad(
  const Eigen::Matrix3d & a,
  const Eigen::Matrix3d & b)
{
  const Eigen::Matrix3d delta = a.transpose() * b;
  const double cos_angle = std::clamp((delta.trace() - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(cos_angle);
}

}  // namespace

BlockRegistrationPipeline::BlockRegistrationPipeline(
  const Eigen::Matrix4d & T_P_C,
  const Eigen::Matrix3d & K,
  const std::vector<TemplateData> & templates,
  const PreprocessingParams & pre,
  const GlobalRegistrationParams & glob,
  const LocalRegistrationParams & loc,
  const rclcpp::Logger & logger,
  bool verbose_logs)
: T_P_C_(T_P_C),
  K_(K),
  templates_(templates),
  pre_(pre),
  glob_(glob),
  loc_(loc),
  logger_(logger),
  verbose_logs_(verbose_logs)
{
}

RegistrationOutput
BlockRegistrationPipeline::run(const RegistrationInput & in)
{
  RegistrationOutput out;

  if (verbose_logs_) {
    RCLCPP_INFO(
      logger_,
      "Registration input: scene_points=%zu",
      in.scene.points_.size());
  }
  out.scene_points = in.scene.points_.size();
  out.used_pose_prior = in.has_pose_prior_world;
  out.prior_position_sigma_m = in.prior_position_sigma_m;
  out.prior_orientation_sigma_rad = in.prior_orientation_sigma_rad;

  geometry::PointCloud cutout;

  if (!computeCutout(in.scene, in.mask, cutout)) {
    RCLCPP_WARN(logger_, "Cutout failed: no points selected from mask.");
    out.failure_stage = "cutout";
    out.failure_reason = "no points selected from mask";
    return out;
  }

  if (verbose_logs_) {
    RCLCPP_INFO(
      logger_,
      "Cutout extracted: points=%zu",
      cutout.points_.size());
  }

  out.debug_mask_cutout = cutout;
  out.debug_mask_cutout.Transform(in.T_world_cloud);
  out.mask_cutout_points = cutout.points_.size();

  preprocess(cutout, in.T_world_cloud);
  out.debug_cleaned_cutout = cutout;
  out.cleaned_cutout_points = cutout.points_.size();

  if (verbose_logs_) {
    RCLCPP_INFO(
      logger_,
      "After preprocess: points=%zu",
      cutout.points_.size());
  }

  if (cutout.points_.empty()) {
    RCLCPP_WARN(logger_, "Preprocess rejected all points.");
    out.failure_stage = "preprocess";
    out.failure_reason = "all points removed during preprocess";
    return out;
  }

  GlobalRegistrationResult glob_res;

  if (in.has_fk_pose_seed) {
    glob_res.success = true;
    glob_res.R_base = in.fk_pose_seed_world.block<3, 3>(0, 0);
    glob_res.center = in.fk_pose_seed_world.block<3, 1>(0, 3);
    glob_res.num_planes = 2;
    if (verbose_logs_) {
      RCLCPP_INFO(logger_, "Global registration bypassed: using FK pose seed.");
    }
  } else {
    glob_res =
      compute_global_registration(
      cutout,
      glob_.Z_WORLD,
      glob_.angle_thresh_top,
      glob_.angle_thresh_front,
      glob_.MAX_PLANES,
      glob_.dist_thresh,
      glob_.min_inliers,
      glob_.max_plane_center_dist,
      glob_.enable_plane_clipping,
      glob_.reject_tall_vertical);

    if (!glob_res.success && glob_.enable_pose_prior_fallback && in.has_pose_prior_world) {
      glob_res.success = true;
      glob_res.R_base = in.pose_prior_world.block<3, 3>(0, 0);
      glob_res.center = in.pose_prior_world.block<3, 1>(0, 3);
      glob_res.num_planes = 2;
      glob_res.plane_cloud = std::make_shared<geometry::PointCloud>(cutout);
      out.used_pose_prior_fallback = true;
      if (!in.has_translation_seed_world) {
        RCLCPP_WARN(
          logger_,
          "Global registration failed; using pose prior fallback without translation seed flag.");
      }
      RCLCPP_WARN(
        logger_,
        "Global registration failed; using pose prior fallback for local ICP seed.");
    } else if (!glob_res.success) {
      RCLCPP_WARN(logger_, "Global registration failed.");
      out.failure_stage = "global_registration";
      out.failure_reason = "global registration failed";
      return out;
    }
  }

  const auto & icp_scene =
    glob_res.plane_cloud ? *glob_res.plane_cloud : cutout;
  if (glob_res.plane_cloud) {
    out.debug_registration_cloud = *glob_res.plane_cloud;
    out.registration_cloud_points = glob_res.plane_cloud->points_.size();
  } else {
    out.registration_cloud_points = cutout.points_.size();
  }

  const Eigen::Vector3d * local_seed_ptr = nullptr;
  if (loc_.use_fk_translation_seed && in.has_translation_seed_world) {
    local_seed_ptr = &in.translation_seed_world;
    if (verbose_logs_) {
      RCLCPP_INFO(
        logger_,
        "Using external translation seed for local ICP: [%.3f %.3f %.3f]",
        in.translation_seed_world.x(),
        in.translation_seed_world.y(),
        in.translation_seed_world.z());
    }
  }

  LocalRegistrationResult reg =
    compute_local_registration(
    icp_scene,
    templates_,
    glob_res,
    loc_.icp_dist,
    loc_.relax_num_faces_match,
    local_seed_ptr,
    loc_.icp_dist_multipliers,
    loc_.enable_point_to_point_fallback);

  if (!reg.success) {
    RCLCPP_WARN(
      logger_,
      "Local ICP refinement failed: reason=%s templates_total=%zu tested=%zu skipped_num_faces=%zu icp_attempts=%zu icp_positive=%zu best_fitness_seen=%.4f best_rmse_seen=%.4f",
      reg.failure_reason.c_str(),
      reg.templates_total,
      reg.templates_tested,
      reg.templates_skipped_num_faces,
      reg.icp_attempts,
      reg.icp_positive,
      reg.best_fitness_seen,
      reg.best_rmse_seen);
    out.failure_stage = "local_icp";
    out.failure_reason = reg.failure_reason;
    return out;
  }

  if (out.used_pose_prior_fallback) {
    const double translation_delta =
      (reg.icp.transformation_.block<3, 1>(0, 3) -
      in.pose_prior_world.block<3, 1>(0, 3)).norm();
    const double orientation_delta =
      rotationDistanceRad(
      in.pose_prior_world.block<3, 3>(0, 0),
      reg.icp.transformation_.block<3, 3>(0, 0));
    if (translation_delta > glob_.pose_prior_fallback_max_translation_m ||
      orientation_delta > glob_.pose_prior_fallback_max_orientation_rad)
    {
      RCLCPP_WARN(
        logger_,
        "Pose-prior fallback rejected: translation_delta=%.3f m orientation_delta=%.3f rad limits=[%.3f m %.3f rad]",
        translation_delta,
        orientation_delta,
        glob_.pose_prior_fallback_max_translation_m,
        glob_.pose_prior_fallback_max_orientation_rad);
      out.failure_stage = "pose_prior_gate";
      out.failure_reason = "pose prior fallback exceeded motion gate";
      return out;
    }
  }

  out.success = true;
  out.T_world_block = reg.icp.transformation_;
  out.fitness = reg.icp.fitness_;
  out.rmse = reg.icp.inlier_rmse_;
  out.template_index = reg.template_index;

  if (verbose_logs_) {
    RCLCPP_INFO(
      logger_,
      "Registration success: template=%d fitness=%.4f rmse=%.4f",
      out.template_index,
      out.fitness,
      out.rmse);
  }

  return out;
}

bool BlockRegistrationPipeline::extractMaskCutout(
  const RegistrationInput & in,
  geometry::PointCloud & cutout_world,
  std::string & reason,
  geometry::PointCloud * mask_cutout_world)
{
  geometry::PointCloud cutout;

  if (!computeCutout(in.scene, in.mask, cutout)) {
    reason = "no points selected from mask";
    RCLCPP_WARN(logger_, "Cutout failed: %s.", reason.c_str());
    return false;
  }

  if (verbose_logs_) {
    RCLCPP_INFO(logger_, "Cutout-only extracted: points=%zu", cutout.points_.size());
  }

  if (mask_cutout_world) {
    *mask_cutout_world = cutout;
    mask_cutout_world->Transform(in.T_world_cloud);
  }

  preprocess(cutout, in.T_world_cloud);
  if (cutout.points_.empty()) {
    reason = "all points removed during preprocess";
    RCLCPP_WARN(logger_, "Cutout-only rejected: %s.", reason.c_str());
    return false;
  }

  cutout_world = std::move(cutout);
  reason = "cutout extracted";
  return true;
}

bool BlockRegistrationPipeline::computeCutout(
  const geometry::PointCloud & scene,
  const cv::Mat & mask,
  geometry::PointCloud & cutout)
{
  auto pts =
    select_points_by_mask(
    scene.points_, mask, K_, T_P_C_);

  if (pts.empty()) {
    return false;
  }

  cutout.points_ = pts;
  cutout.EstimateNormals();
  return true;
}

void BlockRegistrationPipeline::preprocess(
  geometry::PointCloud & cutout,
  const Eigen::Matrix4d & T_world_cloud)
{
  std::shared_ptr<geometry::PointCloud> pcd;
  std::vector<size_t> ind;

  std::tie(pcd, ind) =
    cutout.RemoveStatisticalOutliers(
    pre_.nb_neighbors,
    pre_.std_dev);

  cutout = *pcd;

  if (verbose_logs_) {
    RCLCPP_INFO(
      logger_,
      "After statistical outlier removal: points=%zu",
      cutout.points_.size());
  }

  if (pre_.enable_cluster_filter) {
    keepDominantCluster(cutout);
    if (verbose_logs_) {
      RCLCPP_INFO(
        logger_,
        "After dominant cluster filter: points=%zu",
        cutout.points_.size());
    }
  }

  if (cutout.points_.size() > pre_.max_pts) {
    std::vector<size_t> idx(cutout.points_.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937{42});
    idx.resize(pre_.max_pts);
    cutout = *cutout.SelectByIndex(idx);
    if (verbose_logs_) {
      RCLCPP_INFO(
        logger_,
        "Downsampled to max_pts=%zu (now %zu)",
        pre_.max_pts,
        cutout.points_.size());
    }
  }

  cutout.Transform(T_world_cloud);

  cutout.EstimateNormals(
    geometry::KDTreeSearchParamHybrid(0.02, 30));
}

bool BlockRegistrationPipeline::keepDominantCluster(
  geometry::PointCloud & cutout)
{
  if (cutout.points_.empty()) {
    return false;
  }

  const auto labels =
    cutout.ClusterDBSCAN(
    pre_.cluster_eps,
    pre_.cluster_min_points,
    false);

  if (labels.empty()) {
    RCLCPP_WARN(logger_, "Cluster filter skipped: DBSCAN returned no labels.");
    return false;
  }

  std::unordered_map<int, size_t> counts;
  for (const int label : labels) {
    if (label >= 0) {
      ++counts[label];
    }
  }

  if (counts.empty()) {
    RCLCPP_WARN(logger_, "Cluster filter skipped: no valid cluster found.");
    return false;
  }

  int best_label = -1;
  size_t best_count = 0;
  for (const auto & [label, count] : counts) {
    if (count > best_count) {
      best_label = label;
      best_count = count;
    }
  }

  if (best_label < 0 || best_count < static_cast<size_t>(pre_.cluster_min_size)) {
    RCLCPP_WARN(
      logger_,
      "Cluster filter skipped: dominant cluster too small (size=%zu, min_size=%d).",
      best_count,
      pre_.cluster_min_size);
    return false;
  }

  std::vector<size_t> keep_indices;
  keep_indices.reserve(best_count);
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] == best_label) {
      keep_indices.push_back(i);
    }
  }

  cutout = *cutout.SelectByIndex(keep_indices);

  if (verbose_logs_) {
    RCLCPP_INFO(
      logger_,
      "Cluster filter kept dominant label=%d size=%zu / total=%zu",
      best_label,
      best_count,
      labels.size());
  }

  return true;
}

} // namespace
