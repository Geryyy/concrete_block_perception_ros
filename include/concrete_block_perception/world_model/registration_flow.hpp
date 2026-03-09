#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <sensor_msgs/msg/image.hpp>

#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/utils/world_model_utils.hpp"
#include "ros2_yolos_cpp/srv/segment_image.hpp"

namespace cbp::world_model
{

using DetectionCandidate = std::pair<uint32_t, sensor_msgs::msg::Image>;

std::vector<DetectionCandidate> buildRegistrationCandidates(
  const ros2_yolos_cpp::srv::SegmentImage::Response & seg_res,
  OneShotMode run_mode,
  const std::string & target_block_id);

bool selectBestCandidateByExpectedPose(
  const std::vector<DetectionCandidate> & candidates,
  const concrete_block_perception::msg::Block & expected_target,
  double max_distance_m,
  const std::function<bool(
    uint32_t, const sensor_msgs::msg::Image &, concrete_block_perception::msg::Block &,
    std::string &)> & run_registration,
  concrete_block_perception::msg::Block & out_best_block,
  double & out_best_dist);

}  // namespace cbp::world_model

