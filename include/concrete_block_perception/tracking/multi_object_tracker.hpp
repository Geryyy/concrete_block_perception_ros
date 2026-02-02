#pragma once

#include <vector>

#include <rclcpp/time.hpp>

#include "concrete_block_perception/tracking/measurement.hpp"
#include "concrete_block_perception/tracking/track.hpp"
#include "concrete_block_perception/tracking/tracker_config.hpp"
#include "concrete_block_perception/tracking/logger.hpp"

namespace cbp::tracking
{

class MultiObjectTracker
{
public:
  explicit MultiObjectTracker(const TrackerConfig & config, rclcpp::Logger logger);

  void step(
    const std::vector<Measurement> & measurements,
    const rclcpp::Time & stamp);

  const std::vector<Track> & tracks() const {return tracks_;}

private:
  std::vector<Measurement> deduplicate(
    const std::vector<Measurement> & meas,
    double dist_thresh,
    double iou_thresh);

  // Prediction for all tracks (Δt-aware)
  void predict(const rclcpp::Time & stamp);

  // GNN + KF update + track management
  void associateAndUpdate(
    const std::vector<Measurement> & measurements,
    const rclcpp::Time & stamp);

  // Track lifecycle
  void createTrack(
    const Measurement & meas,
    const rclcpp::Time & stamp);

  void pruneTracks();

private:
  TrackerConfig cfg_;
  rclcpp::Logger logger_;

  std::vector<Track> tracks_;
  int next_track_id_{0};
};

}  // namespace cbp::tracking
