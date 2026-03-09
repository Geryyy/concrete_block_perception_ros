  void resetPerfCounters()
  {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    perf_timing_count_ = 0;
    perf_seg_sum_ms_ = 0;
    perf_track_sum_ms_ = 0;
    perf_reg_sum_ms_ = 0;
    perf_total_sum_ms_ = 0;
    dropped_busy_frames_.store(0);
    dropped_sync_frames_.store(0);
  }

  bool applyPerceptionMode(const std::string & mode)
  {
    cbpwm::PerceptionModeConfig config;
    if (!cbpwm::resolvePerceptionModeConfig(mode, config)) {
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(mode_mutex_);
      perception_mode_ = config.perception_mode;
      pipeline_mode_ = config.pipeline_mode;
      resetPerfCounters();
    }

    RCLCPP_INFO(get_logger(), "%s", config.log_message);
    return true;
  }

  bool needsRegistration() const
  {
    return pipeline_mode_ == cbpwm::PipelineMode::kRegister ||
           pipeline_mode_ == cbpwm::PipelineMode::kFull;
  }

  std::string resolveGraspedBlockId()
  {
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    return cbpwm::resolveGraspedBlockId(persistent_world_, *get_clock());
  }

  cbpwm::AssociationConfig associationConfig() const
  {
    cbpwm::AssociationConfig cfg;
    cfg.association_max_distance_m = association_max_distance_m_;
    cfg.association_max_age_s = association_max_age_s_;
    cfg.min_update_confidence = min_update_confidence_;
    return cfg;
  }

  bool buildCoarseBlockFromMaskAndCloud(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask_msg,
    const sensor_msgs::msg::PointCloud2 & cloud_msg,
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d * camera_origin_world,
    Block & out_block,
    std::string & reason) const
  {
    cbpwm::CoarsePoseInput in;
    in.detection_id = detection_id;
    in.header = header;
    in.mask = toCvMono(mask_msg);
    in.camera_origin_world = camera_origin_world;

    cbpwm::CoarsePoseConfig cfg;
    cfg.min_points = scene_discovery_coarse_fallback_min_points_;
    cfg.square_ratio_thresh = coarse_surface_square_ratio_thresh_;
    cfg.front_center_offset_square_m = coarse_front_center_offset_square_m_;
    cfg.front_center_offset_rect_m = coarse_front_center_offset_rect_m_;
    cfg.min_confidence = std::max(0.3F, static_cast<float>(min_update_confidence_));

    return cbpwm::buildCoarseBlockFromOrganizedCloud(in, cloud_msg, cfg, out_block, reason);
  }

  bool buildCoarseBlockFromCloudCentroid(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask_msg,
    const sensor_msgs::msg::PointCloud2 & cutout_cloud_msg,
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d * camera_origin_world,
    Block & out_block,
    std::string & reason) const
  {
    cbpwm::CoarsePoseInput in;
    in.detection_id = detection_id;
    in.header = header;
    in.mask = toCvMono(mask_msg);
    in.camera_origin_world = camera_origin_world;

    cbpwm::CoarsePoseConfig cfg;
    cfg.min_points = scene_discovery_coarse_fallback_min_points_;
    cfg.square_ratio_thresh = coarse_surface_square_ratio_thresh_;
    cfg.front_center_offset_square_m = coarse_front_center_offset_square_m_;
    cfg.front_center_offset_rect_m = coarse_front_center_offset_rect_m_;
    cfg.min_confidence = std::max(0.3F, static_cast<float>(min_update_confidence_));

    return cbpwm::buildCoarseBlockFromCutoutCloud(in, cutout_cloud_msg, cfg, out_block, reason);
  }

  bool runRegistrationServiceCutoutSync(
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    double timeout_s,
    sensor_msgs::msg::PointCloud2 & out_cutout_cloud,
    std::string & reason)
  {
    if (!register_srv_client_ || !register_srv_client_->service_is_ready()) {
      reason = "register_block_pose service unavailable";
      return false;
    }

    auto req = std::make_shared<RegisterBlockSrv::Request>();
    req->mask = mask;
    req->cloud = cloud;
    req->object_class = object_class_;

    auto future = register_srv_client_->async_send_request(req);
    const auto ret = future.wait_for(std::chrono::duration<double>(timeout_s));
    if (ret != std::future_status::ready) {
      reason = "register_block_pose service timeout";
      return false;
    }
    const auto res = future.get();
    if (!res) {
      reason = "empty register_block_pose response";
      return false;
    }
    if (res->cutout_cloud.data.empty()) {
      reason = "register_block_pose returned empty cutout_cloud";
      return false;
    }
    out_cutout_cloud = res->cutout_cloud;
    reason = "cutout_cloud received from register_block_pose";
    return true;
  }


  bool lookupFrameOriginInWorld(
    const std_msgs::msg::Header & stamped_header,
    Eigen::Vector3d & origin_world,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer unavailable";
      return false;
    }
    std::string source_frame = stamped_header.frame_id;
    if (source_frame.empty()) {
      if (!resolveCameraFrame(stamped_header, source_frame, reason)) {
        return false;
      }
    }
    try {
      const auto tf_w_s = tf_buffer_->lookupTransform(
        world_frame_, source_frame, stamped_header.stamp, tf2::durationFromSec(0.1));
      origin_world =
        Eigen::Vector3d(tf_w_s.transform.translation.x, tf_w_s.transform.translation.y,
        tf_w_s.transform.translation.z);
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed for camera origin (") + world_frame_ + " <- " +
        source_frame + "): " + ex.what();
      return false;
    }
  }

  void resetBusy()
  {
    busy_.store(false);
  }

  void updateLatestWorldCache(const BlockArray & out)
  {
    std::lock_guard<std::mutex> lock(latest_world_mutex_);
    latest_world_ = out;
  }

  BlockArray latestWorldSnapshot()
  {
    std::lock_guard<std::mutex> lock(latest_world_mutex_);
    return latest_world_;
  }

  bool runRegistrationSync(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std_msgs::msg::Header & header,
    double timeout_s,
    Block & out_block,
    std::string & reason,
    const std::string & object_class_override = "")
  {
    RegisterBlock::Goal goal;
    goal.mask = mask;
    goal.cloud = cloud;
    goal.object_class = object_class_override.empty() ? object_class_ : object_class_override;

    std::mutex reg_mutex;
    std::condition_variable reg_cv;
    bool done = false;
    bool goal_rejected = false;
    rclcpp_action::ResultCode result_code = rclcpp_action::ResultCode::UNKNOWN;
    RegisterBlock::Result::SharedPtr action_result;

    rclcpp_action::Client<RegisterBlock>::SendGoalOptions options;
    options.goal_response_callback =
      [&reg_mutex, &reg_cv, &done, &goal_rejected](GoalHandleRegisterBlock::SharedPtr goal_handle)
      {
        if (goal_handle) {
          return;
        }
        std::lock_guard<std::mutex> lock(reg_mutex);
        goal_rejected = true;
        done = true;
        reg_cv.notify_all();
      };
    options.result_callback =
      [&reg_mutex, &reg_cv, &done, &result_code, &action_result](
      const GoalHandleRegisterBlock::WrappedResult & wrapped)
      {
        std::lock_guard<std::mutex> lock(reg_mutex);
        result_code = wrapped.code;
        action_result = wrapped.result;
        done = true;
        reg_cv.notify_all();
      };

    (void)action_client_->async_send_goal(goal, options);

    {
      std::unique_lock<std::mutex> lock(reg_mutex);
      const bool completed = reg_cv.wait_for(
        lock,
        std::chrono::duration<double>(timeout_s),
        [&done]() {return done;});
      if (!completed) {
        reason = "action timeout";
        return false;
      }
      if (goal_rejected) {
        reason = "goal rejected";
        return false;
      }
    }

    if (result_code != rclcpp_action::ResultCode::SUCCEEDED) {
      reason = "result code " + std::to_string(static_cast<int>(result_code));
      return false;
    }

    if (!action_result) {
      reason = "empty result response";
      return false;
    }
    if (!action_result->success) {
      reason = "registration reported success=false";
      return false;
    }
    if (action_result->fitness < min_fitness_ || action_result->rmse > max_rmse_) {
      reason = "thresholds failed (fitness=" + std::to_string(action_result->fitness) +
        ", rmse=" + std::to_string(action_result->rmse) + ")";
      return false;
    }

    out_block.id = "block_" + std::to_string(detection_id);
    out_block.pose = action_result->pose;
    out_block.confidence = action_result->fitness;
    out_block.last_seen = header.stamp;
    out_block.pose_status = Block::POSE_PRECISE;
    out_block.task_status = Block::TASK_FREE;
    return true;
  }

  bool runSegmentationSync(
    const sensor_msgs::msg::Image & image,
    double timeout_s,
    SegmentSrv::Response::SharedPtr & out_response,
    std::string & reason)
  {
    if (!segment_client_ || !segment_client_->service_is_ready()) {
      reason = "segmentation service unavailable";
      return false;
    }

    auto seg_req = std::make_shared<SegmentSrv::Request>();
    seg_req->image = image;
    seg_req->return_debug = false;

    auto future = segment_client_->async_send_request(seg_req);
    const auto ret = future.wait_for(std::chrono::duration<double>(timeout_s));
    if (ret != std::future_status::ready) {
      reason = "segmentation timeout";
      return false;
    }

    out_response = future.get();
    if (!out_response) {
      reason = "empty segmentation response";
      return false;
    }
    if (!out_response->success) {
      reason = "segmentation returned success=false";
      return false;
    }

    return true;
  }

  bool trySceneDiscoveryCoarseFallback(
    uint32_t detection_id,
    const std::string & registration_reason,
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    const cbpwm::SceneFlowRequest & run_request,
    const Eigen::Vector3d * camera_origin_world,
    cbpwm::RegistrationCounters & counters)
  {
    if (run_request.mode != cbpwm::OneShotMode::kSceneDiscovery ||
      !scene_discovery_coarse_fallback_enabled_)
    {
      return false;
    }

    Block coarse_block;
    std::string coarse_reason;
    const bool coarse_ok = buildCoarseBlockFromMaskAndCloud(
      detection_id,
      mask,
      cloud,
      cloud.header,
      camera_origin_world,
      coarse_block,
      coarse_reason);
    if (!coarse_ok) {
      sensor_msgs::msg::PointCloud2 cutout_cloud;
      std::string cutout_reason;
      const bool got_cutout = runRegistrationServiceCutoutSync(
        mask, cloud, run_request.registration_timeout_s, cutout_cloud, cutout_reason);
      if (!got_cutout) {
        RCLCPP_WARN(
          get_logger(),
          "Registration rejected for block_%u (%s). Coarse fallback unavailable: %s",
          detection_id,
          registration_reason.c_str(),
          cutout_reason.c_str());
        return false;
      }

      const bool coarse_from_cutout_ok = buildCoarseBlockFromCloudCentroid(
        detection_id,
        mask,
        cutout_cloud,
        cutout_cloud.header,
        camera_origin_world,
        coarse_block,
        coarse_reason);
      if (!coarse_from_cutout_ok) {
        RCLCPP_WARN(
          get_logger(),
          "Registration rejected for block_%u (%s). Coarse fallback unavailable: %s",
          detection_id,
          registration_reason.c_str(),
          coarse_reason.c_str());
        return false;
      }
    }

    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    std::string assigned_id;
    std::string upsert_reason;
    const bool upsert_ok = cbpwm::upsertRegisteredBlock(
      persistent_world_,
      world_block_counter_,
      coarse_block,
      run_request.mode,
      run_request.target_block_id,
      cloud.header,
      *get_clock(),
      associationConfig(),
      assigned_id,
      upsert_reason);
    if (!upsert_ok) {
      RCLCPP_WARN(
        get_logger(),
        "Registration rejected for block_%u (%s). Coarse fallback rejected: %s",
        detection_id,
        registration_reason.c_str(),
        upsert_reason.c_str());
      return false;
    }

    ++counters.coarse_upserts_ok;
    RCLCPP_WARN(
      get_logger(),
      "Registration rejected for block_%u (%s). Coarse fallback accepted: incoming=%s assigned=%s",
      detection_id,
      registration_reason.c_str(),
      coarse_block.id.c_str(),
      assigned_id.c_str());
    return true;
  }

  

  static Eigen::Matrix4d transformToEigen(const geometry_msgs::msg::TransformStamped & tf)
  {
    Eigen::Quaterniond q(
      tf.transform.rotation.w,
      tf.transform.rotation.x,
      tf.transform.rotation.y,
      tf.transform.rotation.z);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    T(0, 3) = tf.transform.translation.x;
    T(1, 3) = tf.transform.translation.y;
    T(2, 3) = tf.transform.translation.z;
    return T;
  }

  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (msg->k.size() < 9) {
      return;
    }
    CameraIntrinsics intr;
    intr.fx = msg->k[0];
    intr.fy = msg->k[4];
    intr.cx = msg->k[2];
    intr.cy = msg->k[5];
    intr.width = msg->width;
    intr.height = msg->height;
    intr.valid = intr.fx > 0.0 && intr.fy > 0.0;

    if (!intr.valid) {
      return;
    }
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    camera_intrinsics_ = intr;
    camera_info_frame_id_ = msg->header.frame_id;
  }

  bool lookupPredictedGraspedPose(
    const std_msgs::msg::Header & header,
    Eigen::Vector3d & p_world,
    Eigen::Vector3d & p_camera,
    Eigen::Quaterniond & q_world,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer not initialized";
      return false;
    }

    try {
      const auto tf_world_tcp = tf_buffer_->lookupTransform(
        world_frame_,
        refine_grasped_tcp_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));
      std::string camera_frame = refine_grasped_camera_frame_;
      if (camera_frame.empty()) {
        camera_frame = header.frame_id;
      }
      if (camera_frame.empty()) {
        std::lock_guard<std::mutex> lock(camera_info_mutex_);
        camera_frame = camera_info_frame_id_;
      }
      if (camera_frame.empty()) {
        reason = "camera frame unresolved (no override, image frame, or camera_info frame)";
        return false;
      }

      const auto tf_camera_world = tf_buffer_->lookupTransform(
        camera_frame,
        world_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));

      const Eigen::Matrix4d T_world_tcp = transformToEigen(tf_world_tcp);
      const Eigen::Matrix4d T_camera_world = transformToEigen(tf_camera_world);
      const Eigen::Matrix4d T_world_block_pred = T_world_tcp * T_tcp_block_;
      p_world = T_world_block_pred.block<3, 1>(0, 3);
      q_world = Eigen::Quaterniond(T_world_block_pred.block<3, 3>(0, 0)).normalized();
      const Eigen::Vector4d p_block_world_h(
        p_world.x(),
        p_world.y(),
        p_world.z(),
        1.0);
      const Eigen::Vector4d p_block_camera_h = T_camera_world * p_block_world_h;

      p_camera = p_block_camera_h.head<3>();
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed: ") + ex.what();
      return false;
    }
  }

  bool resolveCameraFrame(
    const std_msgs::msg::Header & header,
    std::string & camera_frame,
    std::string & reason)
  {
    camera_frame = refine_grasped_camera_frame_;
    if (camera_frame.empty()) {
      camera_frame = header.frame_id;
    }
    if (camera_frame.empty()) {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      camera_frame = camera_info_frame_id_;
    }
    if (camera_frame.empty()) {
      reason = "camera frame unresolved (no override, image frame, or camera_info frame)";
      return false;
    }
    return true;
  }

  bool worldPointToCamera(
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d & p_world,
    Eigen::Vector3d & p_camera,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer not initialized";
      return false;
    }

    std::string camera_frame;
    if (!resolveCameraFrame(header, camera_frame, reason)) {
      return false;
    }

    try {
      const auto tf_camera_world = tf_buffer_->lookupTransform(
        camera_frame,
        world_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));
      const Eigen::Matrix4d T_camera_world = transformToEigen(tf_camera_world);
      const Eigen::Vector4d p_world_h(p_world.x(), p_world.y(), p_world.z(), 1.0);
      const Eigen::Vector4d p_camera_h = T_camera_world * p_world_h;
      p_camera = p_camera_h.head<3>();
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed: ") + ex.what();
      return false;
    }
  }

  cbpwm::RefineFlowRuntime makeRefineFlowRuntime()
  {
    cbpwm::RefineFlowRuntime rt;
    rt.logger = get_logger();
    rt.registration_ready = [this]() {
        return action_client_ && action_client_->action_server_is_ready();
      };
    rt.publish_persistent_world = [this](const std_msgs::msg::Header & header) {
        publishPersistentWorld(header);
      };
    rt.complete_one_shot = [this](uint64_t sequence, bool success, const std::string & message) {
        completeOneShotRequest(sequence, success, message);
      };
    rt.reset_busy = [this]() {resetBusy();};
    rt.record_timing = [this](int64_t seg_ms, int64_t track_ms, int64_t reg_ms, int64_t total_ms) {
        recordTiming(seg_ms, track_ms, reg_ms, total_ms);
      };
    rt.publish_debug_overlay = [this](const sensor_msgs::msg::Image & image_msg) {
        if (det_debug_pub_) {
          det_debug_pub_->publish(image_msg);
        }
      };
    rt.publish_roi_input = [this](const sensor_msgs::msg::Image & image_msg) {
        if (refine_grasped_roi_input_pub_) {
          refine_grasped_roi_input_pub_->publish(image_msg);
        }
      };
    rt.get_expected_target =
      [this](const std::string & target_id, Block & out_target) {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        const auto it = persistent_world_.find(target_id);
        if (it == persistent_world_.end()) {
          return false;
        }
        out_target = it->second;
        return true;
      };
    rt.get_projection_intrinsics = [this](cbpwm::ProjectionIntrinsics & out_intr) {
        std::lock_guard<std::mutex> lock(camera_info_mutex_);
        out_intr.valid = camera_intrinsics_.valid;
        out_intr.fx = camera_intrinsics_.fx;
        out_intr.fy = camera_intrinsics_.fy;
        out_intr.cx = camera_intrinsics_.cx;
        out_intr.cy = camera_intrinsics_.cy;
        out_intr.width = camera_intrinsics_.width;
        out_intr.height = camera_intrinsics_.height;
        return out_intr.valid;
      };
    rt.lookup_predicted_grasped_pose =
      [this](
      const std_msgs::msg::Header & header,
      Eigen::Vector3d & p_world,
      Eigen::Vector3d & p_camera,
      Eigen::Quaterniond & q_world,
      std::string & reason) {
        return lookupPredictedGraspedPose(header, p_world, p_camera, q_world, reason);
      };
    rt.world_point_to_camera =
      [this](
      const std_msgs::msg::Header & header,
      const Eigen::Vector3d & p_world,
      Eigen::Vector3d & p_camera,
      std::string & reason) {
        return worldPointToCamera(header, p_world, p_camera, reason);
      };
    rt.run_segmentation_sync =
      [this](
      const sensor_msgs::msg::Image & in_image,
      double timeout_s,
      SegmentSrv::Response::SharedPtr & out_response,
      std::string & out_reason) {
        return runSegmentationSync(in_image, timeout_s, out_response, out_reason);
      };
    rt.run_registration_sync =
      [this](
      uint32_t detection_id,
      const sensor_msgs::msg::Image & mask,
      const sensor_msgs::msg::PointCloud2 & cloud,
      const std_msgs::msg::Header & header,
      double timeout_s,
      Block & out_block,
      std::string & out_reason,
      const std::string & object_class_override) {
        return runRegistrationSync(
          detection_id,
          mask,
          cloud,
          header,
          timeout_s,
          out_block,
          out_reason,
          object_class_override);
      };
    rt.upsert_block = [](Block &, std::string &, std::string &) {
        return false;
      };
    return rt;
  }

  void processRefineGraspedWithFkRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start)
  {
    cbpwm::RefineRequest req;
    req.sequence = run_request.sequence;
    req.target_block_id = run_request.target_block_id;
    req.registration_timeout_s = run_request.registration_timeout_s;

    cbpwm::RefineGraspedConfig cfg;
    cfg.roi_cfg.roi_size_x_m = roi_size_x_m_;
    cfg.roi_cfg.roi_size_y_m = roi_size_y_m_;
    cfg.roi_cfg.min_depth_m = refine_grasped_min_depth_m_;
    cfg.roi_cfg.max_depth_m = refine_grasped_max_depth_m_;
    cfg.roi_cfg.segmentation_timeout_s = refine_grasped_segmentation_timeout_s_;
    cfg.roi_cfg.use_black_bg = refine_grasped_use_black_bg_;
    cfg.roi_cfg.blur_kernel_size = refine_grasped_blur_kernel_size_;
    cfg.debug_detection_overlay_enabled = debug_detection_overlay_enabled_;
    cfg.debug_refine_grasped_roi_input_enabled = debug_refine_grasped_roi_input_enabled_;
    cfg.object_class = object_class_;
    cfg.pose_fusion.enabled = refine_grasped_pose_fusion_.enabled;
    cfg.pose_fusion.mode = refine_grasped_pose_fusion_.mode;
    cfg.pose_fusion.max_translation_jump_m = refine_grasped_pose_fusion_.max_translation_jump_m;
    cfg.pose_fusion.max_z_delta_m = refine_grasped_pose_fusion_.max_z_delta_m;
    cfg.pose_fusion.debug_log = refine_grasped_pose_fusion_.debug_log;

    auto rt = makeRefineFlowRuntime();
    rt.upsert_block = [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        return cbpwm::upsertRegisteredBlock(
          persistent_world_,
          world_block_counter_,
          block,
          run_request.mode,
          run_request.target_block_id,
          cloud->header,
          *get_clock(),
          associationConfig(),
          assigned_id,
          reason);
      };

    cbpwm::processRefineGraspedWithFkRoi(req, cfg, rt, image, cloud, t_start);
  }

  bool tryProcessRefineBlockWithPoseRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start)
  {
    cbpwm::RefineRequest req;
    req.sequence = run_request.sequence;
    req.target_block_id = run_request.target_block_id;
    req.registration_timeout_s = run_request.registration_timeout_s;

    cbpwm::RefineBlockConfig cfg;
    cfg.use_pose_roi = refine_block_use_pose_roi_;
    cfg.roi_cfg.roi_size_x_m = refine_block_roi_size_x_m_;
    cfg.roi_cfg.roi_size_y_m = refine_block_roi_size_y_m_;
    cfg.roi_cfg.min_depth_m = refine_block_min_depth_m_;
    cfg.roi_cfg.max_depth_m = refine_block_max_depth_m_;
    cfg.roi_cfg.segmentation_timeout_s = refine_block_segmentation_timeout_s_;
    cfg.roi_cfg.use_black_bg = refine_block_use_black_bg_;
    cfg.roi_cfg.blur_kernel_size = refine_block_blur_kernel_size_;
    cfg.refine_target_max_distance_m = refine_target_max_distance_m_;
    cfg.debug_detection_overlay_enabled = debug_detection_overlay_enabled_;

    auto rt = makeRefineFlowRuntime();
    rt.upsert_block = [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        return cbpwm::upsertRegisteredBlock(
          persistent_world_,
          world_block_counter_,
          block,
          run_request.mode,
          run_request.target_block_id,
          cloud->header,
          *get_clock(),
          associationConfig(),
          assigned_id,
          reason);
      };

    return cbpwm::tryProcessRefineBlockWithPoseRoi(req, cfg, rt, image, cloud, t_start);
  }

