#include "concrete_block_perception/tracking/multi_object_tracker.hpp"

#include <limits>
#include <algorithm>

#include "concrete_block_perception/tracking/gating.hpp"
#include "concrete_block_perception/tracking/hungarian.hpp"
#include "concrete_block_perception/img_utils.hpp"

namespace cbp::tracking
{

MultiObjectTracker::MultiObjectTracker(
  const TrackerConfig & config,
  rclcpp::Logger logger)
: cfg_(config),
  logger_(logger.get_child("tracker"))
{
  TRACKER_LOG(logger_, "MultiObjectTracker initialized");
}

void MultiObjectTracker::step(
  const std::vector<Measurement> & measurements,
  const rclcpp::Time & stamp)
{
  TRACKER_LOG(
    logger_,
    "=== Tracker step @ %.3f | tracks=%zu meas=%zu ===",
    stamp.seconds(),
    tracks_.size(),
    measurements.size());

  predict(stamp);

  auto measurements_clean =
    deduplicate(measurements, cfg_.deduplication_radius, cfg_.iou_thresh);

  associateAndUpdate(measurements_clean, stamp);
  pruneTracks();
}

// --------------------------------------------------
// Measurement deduplication
// --------------------------------------------------
std::vector<Measurement> MultiObjectTracker::deduplicate(
  const std::vector<Measurement> & meas,
  double dist_thresh,
  double iou_thresh)
{
  std::vector<Measurement> out;

  for (const auto & m : meas) {

    bool duplicate = false;

    for (const auto & o : out) {

      const double dist =
        (m.position - o.position).norm();

      double iou = 0.0;
      if (m.bbox.area() > 0 && o.bbox.area() > 0) {
        iou = bboxIoU(m.bbox, o.bbox);
      }

      if (iou > iou_thresh || dist < dist_thresh) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      out.push_back(m);
    }
  }

  return out;
}

// --------------------------------------------------
// Prediction
// --------------------------------------------------
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

// --------------------------------------------------
// Association + Update
// --------------------------------------------------
void MultiObjectTracker::associateAndUpdate(
  const std::vector<Measurement> & measurements,
  const rclcpp::Time & stamp)
{
  const size_t N = tracks_.size();
  const size_t M = measurements.size();

  TRACKER_LOG(
    logger_,
    "Association: N_tracks=%zu M_meas=%zu",
    N, M);

  // --------------------------------------------------
  // No tracks yet
  // --------------------------------------------------
  if (N == 0) {
    for (const auto & m : measurements) {
      createTrack(m, stamp);
    }
    return;
  }

  // --------------------------------------------------
  // No measurements
  // --------------------------------------------------
  if (M == 0) {
    for (auto & t : tracks_) {
      t.misses++;
    }
    return;
  }

  // --------------------------------------------------
  // Cost matrix
  // --------------------------------------------------
  Eigen::MatrixXd cost =
    Eigen::MatrixXd::Constant(
    N, M, std::numeric_limits<double>::infinity());

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {

      const auto & meas = measurements[j];

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

  const auto assignment = Hungarian::solve(cost);
  std::vector<bool> meas_used(M, false);

  // --------------------------------------------------
  // Update tracks
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
      tracks_[i].last_bbox = meas.bbox;

      if (!tracks_[i].confirmed &&
        tracks_[i].hits >= cfg_.min_hits)
      {
        tracks_[i].confirmed = true;
        TRACKER_LOG(
          logger_,
          "Track %d confirmed (hits=%d)",
          tracks_[i].id,
          tracks_[i].hits);
      }

      meas_used[j] = true;

    } else {
      tracks_[i].misses++;
    }
  }

  // --------------------------------------------------
  // Spawn new tracks (with strong suppression)
  // --------------------------------------------------
  for (size_t j = 0; j < M; ++j) {

    if (meas_used[j]) {
      continue;
    }

    // FIX 1: confidence-based birth filtering
    if (measurements[j].confidence < 0.4) {
      TRACKER_LOG(
        logger_,
        "Skipping low-confidence measurement (conf=%.2f)",
        measurements[j].confidence);
      continue;
    }

    bool suppress = false;

    for (const auto & t : tracks_) {

      // FIX 2: suppress near confirmed OR strong tentative tracks
      if (!t.confirmed &&
        t.hits < std::max(1, cfg_.min_hits / 2))
      {
        continue;
      }

      const Eigen::Vector3d dx =
        measurements[j].position - t.kf.x().head<3>();

      if (dx.norm() < cfg_.birth_suppression_radius) {
        suppress = true;
        break;
      }

      // FIX 3: break immediately on IoU suppression
      if (bboxIoU(measurements[j].bbox, t.last_bbox) > cfg_.iou_thresh) {
        suppress = true;
        break;
      }
    }

    if (suppress) {
      TRACKER_LOG(
        logger_,
        "Suppressing birth near existing track");
    } else {
      createTrack(measurements[j], stamp);
    }
  }
}

// --------------------------------------------------
// Track creation
// --------------------------------------------------
void MultiObjectTracker::createTrack(
  const Measurement & meas,
  const rclcpp::Time & stamp)
{
  Track t;
  t.id = next_track_id_++;
  t.age = 1;
  t.hits = 1;
  t.misses = 0;
  t.confirmed = false;
  t.last_update = stamp;
  t.pose_status = Block::POSE_COARSE;
  t.task_status = Block::TASK_FREE;

  Eigen::Vector4d z;
  z << meas.position, meas.yaw;

  t.kf.initialize(z, meas.R);
  t.last_bbox = meas.bbox;

  TRACKER_LOG(
    logger_,
    "Create tentative track %d @ [%.2f %.2f %.2f]",
    t.id,
    meas.position.x(),
    meas.position.y(),
    meas.position.z());

  tracks_.push_back(std::move(t));
}

// --------------------------------------------------
// Pruning
// --------------------------------------------------
void MultiObjectTracker::pruneTracks()
{
  const size_t before = tracks_.size();

  tracks_.erase(
    std::remove_if(
      tracks_.begin(),
      tracks_.end(),
      [&](const Track & t) {
        if (!t.confirmed) {
          return t.misses > 1;   // fast ghost removal
        }
        return t.misses > cfg_.max_misses;
      }),
    tracks_.end());

  const size_t after = tracks_.size();

  if (after != before) {
    TRACKER_LOG_WARN(
      logger_,
      "Pruned %zu tracks (remaining=%zu)",
      before - after, after);
  }
}

}  // namespace cbp::tracking
