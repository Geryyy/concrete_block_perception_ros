#include "concrete_block_perception/registration/block_registration_pipeline.hpp"

using namespace open3d;
using namespace pcd_block;

namespace concrete_block_perception
{

BlockRegistrationPipeline::BlockRegistrationPipeline(
  const Eigen::Matrix4d & T_P_C,
  const Eigen::Matrix3d & K,
  const std::vector<TemplateData> & templates,
  const PreprocessingParams & pre,
  const GlobalRegistrationParams & glob,
  const LocalRegistrationParams & loc)
: T_P_C_(T_P_C),
  K_(K),
  templates_(templates),
  pre_(pre),
  glob_(glob),
  loc_(loc)
{
}

RegistrationOutput
BlockRegistrationPipeline::run(const RegistrationInput & in)
{
  RegistrationOutput out;

  geometry::PointCloud cutout;

  if (!computeCutout(in.scene, in.mask, cutout)) {
    return out;
  }

  preprocess(cutout, in.T_world_cloud);

  GlobalRegistrationResult glob_res =
    compute_global_registration(
    cutout,
    glob_.Z_WORLD,
    glob_.angle_thresh,
    glob_.MAX_PLANES,
    glob_.dist_thresh,
    glob_.min_inliers,
    glob_.max_plane_center_dist,
    glob_.enable_plane_clipping);

  if (!glob_res.success) {
    return out;
  }

  const auto & icp_scene =
    glob_res.plane_cloud ? *glob_res.plane_cloud : cutout;

  LocalRegistrationResult reg =
    compute_local_registration(
    icp_scene,
    templates_,
    glob_res,
    loc_.icp_dist);

  if (!reg.success) {
    return out;
  }

  out.success = true;
  out.T_world_block = reg.icp.transformation_;
  out.fitness = reg.icp.fitness_;
  out.rmse = reg.icp.inlier_rmse_;
  out.template_index = reg.template_index;
  out.debug_scene = icp_scene;

  return out;
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

  if (cutout.points_.size() > pre_.max_pts) {
    std::vector<size_t> idx(cutout.points_.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937{42});
    idx.resize(pre_.max_pts);
    cutout = *cutout.SelectByIndex(idx);
  }

  cutout.Transform(T_world_cloud);

  cutout.EstimateNormals(
    geometry::KDTreeSearchParamHybrid(0.02, 30));
}

} // namespace
