  bool buildRoiMaskFromPrediction(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const Eigen::Vector3d & p_camera,
    const cbpwm::RoiInputConfig & roi_cfg,
    cv::Mat & roi_mask,
    cv::Rect & roi_rect,
    std::string & reason)
  {
    cbpwm::ProjectionIntrinsics intr;
    {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      intr.valid = camera_intrinsics_.valid;
      intr.fx = camera_intrinsics_.fx;
      intr.fy = camera_intrinsics_.fy;
      intr.cx = camera_intrinsics_.cx;
      intr.cy = camera_intrinsics_.cy;
      intr.width = camera_intrinsics_.width;
      intr.height = camera_intrinsics_.height;
    }
    return cbpwm::buildRoiMaskFromPrediction(
      intr, image, p_camera, roi_cfg, roi_mask, roi_rect, reason);
  }

  void processRefineGraspedWithFkRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start)
  {
    if (!action_client_->action_server_is_ready()) {
      completeOneShotRequest(run_request.sequence, false, "Registration action unavailable.");
      resetBusy();
      return;
    }

    Eigen::Vector3d p_world_fk = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_camera = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_world_fk = Eigen::Quaterniond::Identity();
    std::string reason;
    if (!lookupPredictedGraspedPose(image->header, p_world_fk, p_camera, q_world_fk, reason)) {
      RCLCPP_WARN(get_logger(), "REFINE_GRASPED FK+ROI failed: %s", reason.c_str());
      publishPersistentWorld(cloud->header);
      completeOneShotRequest(
        run_request.sequence,
        false,
        "REFINE_GRASPED FK prediction failed: " + reason);
      resetBusy();
      return;
    }

    cv::Mat roi_mask;
    cv::Rect roi_rect;
    cbpwm::RoiInputConfig roi_cfg;
    roi_cfg.roi_size_x_m = roi_size_x_m_;
    roi_cfg.roi_size_y_m = roi_size_y_m_;
    roi_cfg.min_depth_m = refine_grasped_min_depth_m_;
    roi_cfg.max_depth_m = refine_grasped_max_depth_m_;
    roi_cfg.segmentation_timeout_s = refine_grasped_segmentation_timeout_s_;
    roi_cfg.use_black_bg = refine_grasped_use_black_bg_;
    roi_cfg.blur_kernel_size = refine_grasped_blur_kernel_size_;
    if (!buildRoiMaskFromPrediction(image, p_camera, roi_cfg, roi_mask, roi_rect, reason)) {
      RCLCPP_WARN(get_logger(), "REFINE_GRASPED FK+ROI failed: %s", reason.c_str());
      publishPersistentWorld(cloud->header);
      completeOneShotRequest(
        run_request.sequence,
        false,
        "REFINE_GRASPED ROI construction failed: " + reason);
      resetBusy();
      return;
    }

    auto roi_image_msg = cbpwm::buildRoiSegmentationInputImage(image, roi_rect, roi_cfg);

    if (debug_refine_grasped_roi_input_enabled_ && refine_grasped_roi_input_pub_) {
      refine_grasped_roi_input_pub_->publish(*roi_image_msg);
    }

    cv::Mat full_seg_mask;
    size_t detections_count = 0;
    if (!cbpwm::roiSegmentationToFullMask(
        image,
        roi_mask,
        roi_image_msg,
        roi_cfg,
        [this](
          const sensor_msgs::msg::Image & in_image,
          double timeout_s,
          SegmentSrv::Response::SharedPtr & out_response,
          std::string & out_reason) {
          return runSegmentationSync(in_image, timeout_s, out_response, out_reason);
        },
        full_seg_mask,
        detections_count,
        reason))
    {
      RCLCPP_WARN(get_logger(), "REFINE_GRASPED FK+ROI segmentation failed: %s", reason.c_str());
      publishPersistentWorld(cloud->header);
      completeOneShotRequest(
        run_request.sequence,
        false,
        "REFINE_GRASPED ROI segmentation failed: " + reason);
      resetBusy();
      return;
    }

    if (debug_detection_overlay_enabled_ && det_debug_pub_) {
      if (auto dbg = cbpwm::buildRoiSegmentationDebugOverlay(image, roi_rect, full_seg_mask)) {
        det_debug_pub_->publish(*dbg);
      }
    }

    RCLCPP_INFO(
      get_logger(),
      "REFINE_GRASPED FK+ROI segmentation: roi=[x=%d y=%d w=%d h=%d] detections=%zu mask_pixels=%d",
      roi_rect.x,
      roi_rect.y,
      roi_rect.width,
      roi_rect.height,
      detections_count,
      cv::countNonZero(full_seg_mask));

    auto full_mask_msg = cv_bridge::CvImage(image->header, "mono8", full_seg_mask).toImageMsg();
    Block block;
    bool registration_ok = false;
    std::string reg_reason;
    const auto t_reg_start = std::chrono::steady_clock::now();
    registration_ok = runRegistrationSync(
      1U,
      *full_mask_msg,
      *cloud,
      cloud->header,
      run_request.registration_timeout_s,
      block,
      reg_reason,
      object_class_ + "#REFINE_GRASPED");
    const auto t_reg_end = std::chrono::steady_clock::now();

    size_t registrations_ok = 0;
    if (registration_ok) {
      bool fusion_ok = true;
      std::string fusion_reason;

      if (refine_grasped_pose_fusion_.enabled) {
        if (refine_grasped_pose_fusion_.mode != "position_from_registration_orientation_from_fk") {
          fusion_ok = false;
          fusion_reason = "unsupported pose_fusion.mode='" + refine_grasped_pose_fusion_.mode + "'";
        } else {
          const auto fusion_result = cbpwm::fuseRegistrationPositionWithFkOrientation(
            block.pose,
            p_world_fk,
            q_world_fk,
            refine_grasped_pose_fusion_.max_translation_jump_m,
            refine_grasped_pose_fusion_.max_z_delta_m);
          fusion_ok = fusion_result.success;
          fusion_reason = fusion_result.reason;

          if (refine_grasped_pose_fusion_.debug_log) {
            const Eigen::Vector3d p_reg(
              block.pose.position.x,
              block.pose.position.y,
              block.pose.position.z);
            RCLCPP_INFO(
              get_logger(),
              "REFINE_GRASPED pose fusion: source=FK_ORIENTATION fk_pos=[%.3f %.3f %.3f] reg_pos=[%.3f %.3f %.3f] fused_pos=[%.3f %.3f %.3f] residual=%.3f z_delta=%.3f gate=%s",
              p_world_fk.x(), p_world_fk.y(), p_world_fk.z(),
              p_reg.x(), p_reg.y(), p_reg.z(),
              block.pose.position.x, block.pose.position.y, block.pose.position.z,
              fusion_result.residual_norm,
              fusion_result.z_delta,
              fusion_ok ? "PASS" : "FAIL");
          }
        }
      }

      if (!fusion_ok) {
        RCLCPP_WARN(get_logger(), "REFINE_GRASPED pose fusion rejected: %s", fusion_reason.c_str());
        reg_reason = fusion_reason;
      }

      if (!fusion_ok) {
        // keep world model unchanged on fusion gate failure
      } else {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        std::string assigned_id;
        std::string upsert_reason;
        const bool upsert_ok = cbpwm::upsertRegisteredBlock(
          persistent_world_,
          world_block_counter_,
          block,
          run_request.mode,
          run_request.target_block_id,
          cloud->header,
          *get_clock(),
          associationConfig(),
          assigned_id,
          upsert_reason);
        if (upsert_ok) {
          ++registrations_ok;
          RCLCPP_INFO(
            get_logger(),
            "REFINE_GRASPED FK+ROI accepted: target=%s assigned=%s roi=[x=%d y=%d w=%d h=%d]",
            run_request.target_block_id.c_str(),
            assigned_id.c_str(),
            roi_rect.x,
            roi_rect.y,
            roi_rect.width,
            roi_rect.height);
        } else {
          RCLCPP_WARN(
            get_logger(),
            "REFINE_GRASPED FK+ROI rejected after association checks: %s",
            upsert_reason.c_str());
          reg_reason = upsert_reason;
        }
      }
    } else {
      RCLCPP_WARN(
        get_logger(),
        "REFINE_GRASPED FK+ROI registration failed: %s",
        reg_reason.c_str());
    }

    const auto reg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_reg_end - t_reg_start).count();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_reg_end - t_start).count();
    recordTiming(0, 0, reg_ms, total_ms);

    publishPersistentWorld(cloud->header);
    completeOneShotRequest(
      run_request.sequence,
      registrations_ok > 0,
      registrations_ok > 0 ?
      ("Pose estimation finished (mode=REFINE_GRASPED_FK_ROI, registration_candidates=1, registrations=1).") :
      ("Pose estimation finished (mode=REFINE_GRASPED_FK_ROI, registration_candidates=1, registrations=0, reason=" +
      reg_reason + ")."));
    resetBusy();
  }

  bool tryProcessRefineBlockWithPoseRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start)
  {
    if (!refine_block_use_pose_roi_ || run_request.mode != cbpwm::OneShotMode::kRefineBlock) {
      return false;
    }
    if (run_request.target_block_id.empty()) {
      return false;
    }
    if (!action_client_->action_server_is_ready()) {
      completeOneShotRequest(run_request.sequence, false, "Registration action unavailable.");
      resetBusy();
      return true;
    }

    Block expected_target;
    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      const auto it = persistent_world_.find(run_request.target_block_id);
      if (it == persistent_world_.end()) {
        RCLCPP_WARN(
          get_logger(),
          "REFINE_BLOCK pose+ROI fallback: target '%s' not found in world model.",
          run_request.target_block_id.c_str());
        return false;
      }
      expected_target = it->second;
    }

    cbpwm::RoiInputConfig roi_cfg;
    roi_cfg.roi_size_x_m = refine_block_roi_size_x_m_;
    roi_cfg.roi_size_y_m = refine_block_roi_size_y_m_;
    roi_cfg.min_depth_m = refine_block_min_depth_m_;
    roi_cfg.max_depth_m = refine_block_max_depth_m_;
    roi_cfg.segmentation_timeout_s = refine_block_segmentation_timeout_s_;
    roi_cfg.use_black_bg = refine_block_use_black_bg_;
    roi_cfg.blur_kernel_size = refine_block_blur_kernel_size_;

    const Eigen::Vector3d p_world(
      expected_target.pose.position.x,
      expected_target.pose.position.y,
      expected_target.pose.position.z);
    Eigen::Vector3d p_camera = Eigen::Vector3d::Zero();
    std::string reason;
    if (!worldPointToCamera(image->header, p_world, p_camera, reason)) {
      RCLCPP_WARN(get_logger(), "REFINE_BLOCK pose+ROI fallback: %s", reason.c_str());
      return false;
    }

    cv::Mat roi_mask;
    cv::Rect roi_rect;
    if (!buildRoiMaskFromPrediction(image, p_camera, roi_cfg, roi_mask, roi_rect, reason)) {
      RCLCPP_WARN(get_logger(), "REFINE_BLOCK pose+ROI fallback: %s", reason.c_str());
      return false;
    }

    auto roi_image_msg = cbpwm::buildRoiSegmentationInputImage(image, roi_rect, roi_cfg);
    cv::Mat full_seg_mask;
    size_t detections_count = 0;
    if (!cbpwm::roiSegmentationToFullMask(
        image,
        roi_mask,
        roi_image_msg,
        roi_cfg,
        [this](
          const sensor_msgs::msg::Image & in_image,
          double timeout_s,
          SegmentSrv::Response::SharedPtr & out_response,
          std::string & out_reason) {
          return runSegmentationSync(in_image, timeout_s, out_response, out_reason);
        },
        full_seg_mask,
        detections_count,
        reason))
    {
      RCLCPP_WARN(get_logger(), "REFINE_BLOCK pose+ROI failed: %s", reason.c_str());
      publishPersistentWorld(cloud->header);
      completeOneShotRequest(
        run_request.sequence,
        false,
        "REFINE_BLOCK ROI segmentation failed: " + reason);
      resetBusy();
      return true;
    }

    if (debug_detection_overlay_enabled_ && det_debug_pub_) {
      if (auto dbg = cbpwm::buildRoiSegmentationDebugOverlay(image, roi_rect, full_seg_mask)) {
        det_debug_pub_->publish(*dbg);
      }
    }

    RCLCPP_INFO(
      get_logger(),
      "REFINE_BLOCK pose+ROI: target=%s roi=[x=%d y=%d w=%d h=%d] detections=%zu mask_pixels=%d",
      run_request.target_block_id.c_str(),
      roi_rect.x,
      roi_rect.y,
      roi_rect.width,
      roi_rect.height,
      detections_count,
      cv::countNonZero(full_seg_mask));

    auto full_mask_msg = cv_bridge::CvImage(image->header, "mono8", full_seg_mask).toImageMsg();
    Block block;
    std::string reg_reason;
    const auto t_reg_start = std::chrono::steady_clock::now();
    const bool registration_ok = runRegistrationSync(
      1U,
      *full_mask_msg,
      *cloud,
      cloud->header,
      run_request.registration_timeout_s,
      block,
      reg_reason);
    const auto t_reg_end = std::chrono::steady_clock::now();

    size_t registrations_ok = 0;
    if (registration_ok) {
      const double dist = cbpwm::blockDistance(block, expected_target);
      if (dist > refine_target_max_distance_m_) {
        reg_reason = "registered pose too far from target: dist=" + std::to_string(dist) + " m";
        RCLCPP_WARN(get_logger(), "REFINE_BLOCK pose+ROI rejected: %s", reg_reason.c_str());
      } else {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        std::string assigned_id;
        std::string upsert_reason;
        const bool upsert_ok = cbpwm::upsertRegisteredBlock(
          persistent_world_,
          world_block_counter_,
          block,
          run_request.mode,
          run_request.target_block_id,
          cloud->header,
          *get_clock(),
          associationConfig(),
          assigned_id,
          upsert_reason);
        if (upsert_ok) {
          ++registrations_ok;
          RCLCPP_INFO(
            get_logger(),
            "REFINE_BLOCK pose+ROI accepted: target=%s assigned=%s dist=%.3f m",
            run_request.target_block_id.c_str(),
            assigned_id.c_str(),
            dist);
        } else {
          reg_reason = upsert_reason;
          RCLCPP_WARN(get_logger(), "REFINE_BLOCK pose+ROI rejected: %s", reg_reason.c_str());
        }
      }
    } else {
      RCLCPP_WARN(get_logger(), "REFINE_BLOCK pose+ROI registration failed: %s", reg_reason.c_str());
    }

    const auto reg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_reg_end - t_reg_start).count();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_reg_end - t_start).count();
    recordTiming(0, 0, reg_ms, total_ms);

    publishPersistentWorld(cloud->header);
    completeOneShotRequest(
      run_request.sequence,
      registrations_ok > 0,
      registrations_ok > 0 ?
      ("Pose estimation finished (mode=REFINE_BLOCK_POSE_ROI, registration_candidates=1, registrations=1).") :
      ("Pose estimation finished (mode=REFINE_BLOCK_POSE_ROI, registration_candidates=1, registrations=0, reason=" +
      reg_reason + ")."));
    resetBusy();
    return true;
  }
