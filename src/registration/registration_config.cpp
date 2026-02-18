#include "concrete_block_perception/registration/registration_config.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
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

  node.declare_parameter<double>("glob_reg.dist_thresh", 0.02);
  node.declare_parameter<int>("glob_reg.min_inliers", 100);
  node.declare_parameter<double>("glob_reg.angle_thresh_degree", 30.0);
  node.declare_parameter<double>("glob_reg.max_plane_center_dist", 0.6);
  node.declare_parameter<bool>("glob_reg.enable_plane_clipping", false);

  node.declare_parameter<double>("loc_reg.icp_dist", 0.04);

  node.declare_parameter<bool>("debug.publish_cutout", true);
  node.declare_parameter<bool>("debug.publish_mask", true);

  node.declare_parameter<bool>("dump.enable", false);
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

  // ------------------------------------------------------------
  // Local registration params
  // ------------------------------------------------------------

  cfg.local.icp_dist =
    node.get_parameter("loc_reg.icp_dist").as_double();

  // ------------------------------------------------------------
  // Debug + dump
  // ------------------------------------------------------------

  cfg.publish_debug_cutout =
    node.get_parameter("debug.publish_cutout").as_bool();

  cfg.publish_debug_mask =
    node.get_parameter("debug.publish_mask").as_bool();

  cfg.dump_enabled =
    node.get_parameter("dump.enable").as_bool();

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

  return cfg;
}

} // namespace
