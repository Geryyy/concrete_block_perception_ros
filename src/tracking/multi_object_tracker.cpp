#include "concrete_block_perception/tracking/multi_object_tracker.hpp"

#include <limits>

#include "concrete_block_perception/tracking/gating.hpp"
#include "concrete_block_perception/tracking/hungarian.hpp"

namespace cbp::tracking
{

MultiObjectTracker::MultiObjectTracker(const TrackerConfig & config)
: cfg_(config)
{
}

void MultiObjectTracker::step(
  const std::vector<Measurement> & measurements,
  const rclcpp::Time & stamp)
{
  predict(stamp);
  associateAndUpdate(measurements, stamp);
  pruneTracks();
}

void MultiObjectTracker::predict(const rclcpp::Time & stamp)
{
  for (auto & track : tracks_) {

    if (track.last_update.nanoseconds() == 0) {
      track.last_update = stamp;
      continue;
    }

    const double dt =
      (stamp - track.last_update).seconds();

    track.kf.predict(
      dt,
      cfg_.Q,
      cfg_.velocity_damping);

    track.age++;
  }
}

void MultiObjectTracker::associateAndUpdate(
  const std::vector<Measurement> & measurements,
  const rclcpp::Time & stamp)
{
  const size_t N = tracks_.size();
  const size_t M = measurements.size();

  // --------------------------------------------------
  // No existing tracks → spawn from all measurements
  // --------------------------------------------------
  if (N == 0) {
    for (const auto & m : measurements) {
      createTrack(m, stamp);
    }
    return;
  }

  // --------------------------------------------------
  // No measurements → mark all tracks missed
  // --------------------------------------------------
  if (M == 0) {
    for (auto & t : tracks_) {
      t.misses++;
    }
    return;
  }

  // --------------------------------------------------
  // Build cost matrix (Mahalanobis, gated)
  // --------------------------------------------------
  Eigen::MatrixXd cost =
    Eigen::MatrixXd::Constant(
    N, M, std::numeric_limits<double>::infinity());

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {

      const auto & meas = measurements[j];

      // Gate on POSITION ONLY (x,y,z)
      auto gate = Gating::gatePosition3D(
        tracks_[i].kf.x(),
        tracks_[i].kf.P(),
        meas.position,
        meas.R.block<3, 3>(0, 0),
        cfg_.chi2_gate);

      if (gate.accepted) {
        cost(i, j) = gate.mahalanobis_sq;
      }
    }
  }

  // --------------------------------------------------
  // Global nearest-neighbor assignment
  // --------------------------------------------------
  const auto assignment = Hungarian::solve(cost);

  std::vector<bool> meas_used(M, false);

  // --------------------------------------------------
  // Update assigned tracks
  // --------------------------------------------------
  for (size_t i = 0; i < assignment.size(); ++i) {

    const int j = assignment[i];

    if (j >= 0) {
      const auto & meas = measurements[j];

      Eigen::Vector4d z;
      z << meas.position, meas.yaw;

      tracks_[i].kf.update(z, meas.R);

      tracks_[i].hits++;
      tracks_[i].misses = 0;
      tracks_[i].last_update = stamp;

      meas_used[j] = true;
    } else {
      tracks_[i].misses++;
    }
  }

  // --------------------------------------------------
  // Spawn new tracks from unassigned measurements
  // --------------------------------------------------
  for (size_t j = 0; j < M; ++j) {
    if (!meas_used[j]) {
      createTrack(measurements[j], stamp);
    }
  }
}

void MultiObjectTracker::createTrack(
  const Measurement & meas,
  const rclcpp::Time & stamp)
{
  Track t;
  t.id = next_track_id_++;
  t.age = 1;
  t.hits = 1;
  t.misses = 0;
  t.last_update = stamp;

  Eigen::Vector4d z;
  z << meas.position, meas.yaw;

  t.kf.initialize(z, meas.R);

  tracks_.push_back(std::move(t));
}

void MultiObjectTracker::pruneTracks()
{
  tracks_.erase(
    std::remove_if(
      tracks_.begin(),
      tracks_.end(),
      [&](const Track & t) {
        return t.misses > cfg_.max_misses;
      }),
    tracks_.end());
}

}  // namespace cbp::tracking
