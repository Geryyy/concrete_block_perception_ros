#include "concrete_block_perception/registration/registration_config.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <filesystem>
#include <cmath>

#include "pcd_block_estimation/yaml_utils.hpp"
#include "pcd_block_estimation/template_utils.hpp"

using namespace pcd_block;

namespace concrete_block_perception
{

BlockRegistrationConfig
load_registration_config(rclcpp::Node & node)
{
  BlockRegistrationConfig cfg;

  // ------------------------------------------------------------
  // Paths
  // ------------------------------------------------------------
  const std::string pkg =
    ament_index_cpp::get_package_share_directory(
    "concrete_block_perception");

  const std::string config_dir = pkg + "/config";

  // ------------------------------------------------------------
  // Declare parameters
  // ------------------------------------------------------------

  node.declare_parameter<std::string>("world_frame", "world");
  node.declare_parameter<std::string>(
    "calib_yaml",
    "calib_zed2i_to_seyond.yaml");

  node.declare_parameter<std::string>("template.dir", "templates");
  node.declare_parameter<std::string>(
    "template.cad_name",
    "ConcreteBlock.ply");
  node.declare_parameter<int>("template.n_points", 2000);
  node.declare_parameter<double>("template.angle_deg", 15.0);

  node.declare_parameter<int>("preproc.max_pts", 500);
  node.declare_parameter<int>("preproc.nb_neighbors", 20);
  node.declare_parameter<double>("preproc.std_dev", 2.0);
  node.declare_parameter<bool>("preproc.enable_cluster_filter", false);
  node.declare_parameter<double>("preproc.cluster_eps", 0.08);
  node.declare_parameter<int>("preproc.cluster_min_points", 20);
  node.declare_parameter<int>("preproc.cluster_min_size", 100);

  node.declare_parameter<double>("glob_reg.dist_thresh", 0.02);
  node.declare_parameter<int>("glob_reg.min_inliers", 100);
  node.declare_parameter<double>("glob_reg.angle_thresh_degree", 30.0);
  node.declare_parameter<double>("glob_reg.max_plane_center_dist", 0.6);
  node.declare_parameter<bool>("glob_reg.enable_plane_clipping", false);
  node.declare_parameter<bool>("glob_reg.reject_tall_vertical", true);

  node.declare_parameter<double>("loc_reg.icp_dist", 0.04);
  node.declare_parameter<bool>("loc_reg.relax_num_faces_match", false);
  node.declare_parameter<bool>("loc_reg.use_fk_translation_seed", false);
  node.declare_parameter<std::vector<double>>("loc_reg.icp_dist_multipliers", {1.0, 1.5, 2.0});
  node.declare_parameter<bool>("loc_reg.enable_point_to_point_fallback", true);
  node.declare_parameter<std::string>("loc_reg.fk_seed.tcp_frame", "elastic/K8_tool_center_point");
  node.declare_parameter<std::vector<double>>("loc_reg.fk_seed.tcp_to_block_xyz", {0.0, 0.0, 0.0});

  node.declare_parameter<double>("teaser_reg.noise_bound", 0.02);
  node.declare_parameter<double>("teaser_reg.cbar2", 1.0);
  node.declare_parameter<bool>("teaser_reg.estimate_scaling", false);
  node.declare_parameter<double>("teaser_reg.rotation_gnc_factor", 1.4);
  node.declare_parameter<int>("teaser_reg.rotation_max_iterations", 100);
  node.declare_parameter<double>("teaser_reg.rotation_cost_threshold", 1e-6);
  node.declare_parameter<double>("teaser_reg.max_clique_time_limit_s", 0.2);
  node.declare_parameter<int>("teaser_reg.min_correspondences", 30);
  node.declare_parameter<int>("teaser_reg.max_template_points", 1000);
  node.declare_parameter<double>("teaser_reg.nn_corr_max_dist", 0.08);
  node.declare_parameter<bool>("teaser_reg.enable_icp_refinement", true);
  node.declare_parameter<double>("teaser_reg.icp_refine_dist", 0.04);
  node.declare_parameter<double>("teaser_reg.eval_corr_dist", 0.04);

  node.declare_parameter<bool>("debug.publish_cutout", true);
  node.declare_parameter<bool>("debug.publish_mask", true);
  node.declare_parameter<bool>("debug.verbose_logs", true);

  node.declare_parameter<bool>("dump.enable", false);
  node.declare_parameter<bool>("dump.failure_package", true);
  node.declare_parameter<std::string>("dump.dir", "dump");

  // ------------------------------------------------------------
  // Read parameters
  // ------------------------------------------------------------

  cfg.world_frame =
    node.get_parameter("world_frame").as_string();

  const std::string calib_yaml_name =
    node.get_parameter("calib_yaml").as_string();

  const std::string calib_path =
    config_dir + "/" + calib_yaml_name;

  if (!std::filesystem::exists(calib_path)) {
    throw std::runtime_error(
            "Calibration YAML not found: " + calib_path);
  }

  cfg.T_P_C = load_T_4x4(calib_path);
  cfg.K = load_camera_matrix(calib_path);

  // ------------------------------------------------------------
  // Template generation
  // ------------------------------------------------------------

  TemplateGenerationParams tpl_params;

  const std::string template_dir_name =
    node.get_parameter("template.dir").as_string();

  const std::string template_dir =
    config_dir + "/" + template_dir_name;

  tpl_params.n_points =
    node.get_parameter("template.n_points").as_int();

  tpl_params.angle_deg =
    node.get_parameter("template.angle_deg").as_double();

  tpl_params.cad_path =
    config_dir + "/" +
    node.get_parameter("template.cad_name").as_string();

  tpl_params.out_dir = template_dir;

  if (!std::filesystem::exists(tpl_params.out_dir)) {
    RCLCPP_INFO(
      node.get_logger(),
      "Generating templates from %s",
      tpl_params.cad_path.c_str());

    generate_templates(tpl_params);
  }

  cfg.templates = load_templates(tpl_params.out_dir);

  // ------------------------------------------------------------
  // Preprocessing params
  // ------------------------------------------------------------

  cfg.preproc.max_pts =
    node.get_parameter("preproc.max_pts").as_int();

  cfg.preproc.nb_neighbors =
    node.get_parameter("preproc.nb_neighbors").as_int();

  cfg.preproc.std_dev =
    node.get_parameter("preproc.std_dev").as_double();
  cfg.preproc.enable_cluster_filter =
    node.get_parameter("preproc.enable_cluster_filter").as_bool();
  cfg.preproc.cluster_eps =
    node.get_parameter("preproc.cluster_eps").as_double();
  cfg.preproc.cluster_min_points =
    node.get_parameter("preproc.cluster_min_points").as_int();
  cfg.preproc.cluster_min_size =
    node.get_parameter("preproc.cluster_min_size").as_int();

  // ------------------------------------------------------------
  // Global registration params
  // ------------------------------------------------------------

  cfg.glob.dist_thresh =
    node.get_parameter("glob_reg.dist_thresh").as_double();

  cfg.glob.min_inliers =
    node.get_parameter("glob_reg.min_inliers").as_int();

  cfg.glob.max_plane_center_dist =
    node.get_parameter("glob_reg.max_plane_center_dist").as_double();

  double angle_deg =
    node.get_parameter("glob_reg.angle_thresh_degree").as_double();

  cfg.glob.angle_thresh =
    std::cos(angle_deg * M_PI / 180.0);

  bool enable_plane_clipping =
    node.get_parameter("glob_reg.enable_plane_clipping").as_bool();
  cfg.glob.enable_plane_clipping = enable_plane_clipping;
  cfg.glob.reject_tall_vertical =
    node.get_parameter("glob_reg.reject_tall_vertical").as_bool();

  // ------------------------------------------------------------
  // Local registration params
  // ------------------------------------------------------------

  cfg.local.icp_dist =
    node.get_parameter("loc_reg.icp_dist").as_double();
  cfg.local.relax_num_faces_match =
    node.get_parameter("loc_reg.relax_num_faces_match").as_bool();
  cfg.local.use_fk_translation_seed =
    node.get_parameter("loc_reg.use_fk_translation_seed").as_bool();
  cfg.local.icp_dist_multipliers =
    node.get_parameter("loc_reg.icp_dist_multipliers").as_double_array();
  if (cfg.local.icp_dist_multipliers.empty()) {
    cfg.local.icp_dist_multipliers = {1.0};
  }
  cfg.local.enable_point_to_point_fallback =
    node.get_parameter("loc_reg.enable_point_to_point_fallback").as_bool();
  cfg.fk_seed_tcp_frame =
    node.get_parameter("loc_reg.fk_seed.tcp_frame").as_string();
  const auto fk_seed_tcp_to_block_xyz =
    node.get_parameter("loc_reg.fk_seed.tcp_to_block_xyz").as_double_array();
  if (fk_seed_tcp_to_block_xyz.size() >= 3) {
    cfg.fk_seed_tcp_to_block_xyz =
      Eigen::Vector3d(
      fk_seed_tcp_to_block_xyz[0],
      fk_seed_tcp_to_block_xyz[1],
      fk_seed_tcp_to_block_xyz[2]);
  }

  cfg.teaser.noise_bound =
    node.get_parameter("teaser_reg.noise_bound").as_double();
  cfg.teaser.cbar2 =
    node.get_parameter("teaser_reg.cbar2").as_double();
  cfg.teaser.estimate_scaling =
    node.get_parameter("teaser_reg.estimate_scaling").as_bool();
  cfg.teaser.rotation_gnc_factor =
    node.get_parameter("teaser_reg.rotation_gnc_factor").as_double();
  cfg.teaser.rotation_max_iterations =
    node.get_parameter("teaser_reg.rotation_max_iterations").as_int();
  cfg.teaser.rotation_cost_threshold =
    node.get_parameter("teaser_reg.rotation_cost_threshold").as_double();
  cfg.teaser.max_clique_time_limit_s =
    node.get_parameter("teaser_reg.max_clique_time_limit_s").as_double();
  const int teaser_min_corr =
    static_cast<int>(node.get_parameter("teaser_reg.min_correspondences").as_int());
  const int teaser_max_tpl_pts =
    static_cast<int>(node.get_parameter("teaser_reg.max_template_points").as_int());
  cfg.teaser.min_correspondences = static_cast<size_t>(std::max(3, teaser_min_corr));
  cfg.teaser.max_template_points = static_cast<size_t>(std::max(16, teaser_max_tpl_pts));
  cfg.teaser.nn_corr_max_dist =
    node.get_parameter("teaser_reg.nn_corr_max_dist").as_double();
  cfg.teaser.enable_icp_refinement =
    node.get_parameter("teaser_reg.enable_icp_refinement").as_bool();
  cfg.teaser.icp_refine_dist =
    node.get_parameter("teaser_reg.icp_refine_dist").as_double();
  cfg.teaser.eval_corr_dist =
    node.get_parameter("teaser_reg.eval_corr_dist").as_double();

  // ------------------------------------------------------------
  // Debug + dump
  // ------------------------------------------------------------

  cfg.publish_debug_cutout =
    node.get_parameter("debug.publish_cutout").as_bool();

  cfg.publish_debug_mask =
    node.get_parameter("debug.publish_mask").as_bool();
  cfg.verbose_logs =
    node.get_parameter("debug.verbose_logs").as_bool();

  cfg.dump_enabled =
    node.get_parameter("dump.enable").as_bool();
  cfg.dump_failure_package =
    node.get_parameter("dump.failure_package").as_bool();

  const std::string dump_dir_rel =
    node.get_parameter("dump.dir").as_string();

  cfg.dump_dir =
    config_dir + "/" + dump_dir_rel;

  if (cfg.dump_enabled) {
    std::filesystem::create_directories(cfg.dump_dir);

    RCLCPP_WARN(
      node.get_logger(),
      "Dump ENABLED → writing to %s",
      cfg.dump_dir.c_str());
  }

  RCLCPP_INFO(
    node.get_logger(),
    "Block registration config loaded");
  RCLCPP_INFO(
    node.get_logger(),
    "  calib: %s",
    calib_path.c_str());
  RCLCPP_INFO(
    node.get_logger(),
    "  templates: %s",
    tpl_params.out_dir.c_str());
  RCLCPP_INFO(
    node.get_logger(),
    "  loc_reg: icp_dist=%.3f relax_num_faces_match=%s use_fk_translation_seed=%s p2p_fallback=%s multipliers=%zu",
    cfg.local.icp_dist,
    cfg.local.relax_num_faces_match ? "true" : "false",
    cfg.local.use_fk_translation_seed ? "true" : "false",
    cfg.local.enable_point_to_point_fallback ? "true" : "false",
    cfg.local.icp_dist_multipliers.size());
  RCLCPP_INFO(
    node.get_logger(),
    "  teaser_reg: noise_bound=%.3f min_corr=%zu nn_max_dist=%.3f icp_refine=%s",
    cfg.teaser.noise_bound,
    cfg.teaser.min_correspondences,
    cfg.teaser.nn_corr_max_dist,
    cfg.teaser.enable_icp_refinement ? "true" : "false");

  return cfg;
}

} // namespace
