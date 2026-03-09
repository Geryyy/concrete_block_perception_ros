  void completeOneShotRequest(uint64_t sequence, bool success, const std::string & message)
  {
    std::lock_guard<std::mutex> lock(one_shot_mutex_);
    if (sequence == 0 || sequence != active_one_shot_.sequence) {
      return;
    }
    active_one_shot_ = OneShotRequest{};
    one_shot_done_sequence_ = sequence;
    one_shot_last_success_ = success;
    one_shot_last_message_ = message;
    one_shot_cv_.notify_all();
    (void)applyPerceptionMode("IDLE");
  }

  void handleRunPoseEstimation(
    const std::shared_ptr<RunPoseSrv::Request> request,
    std::shared_ptr<RunPoseSrv::Response> response)
  {
    const cbpwm::OneShotMode run_mode = cbpwm::parseOneShotMode(request->mode);
    if (run_mode == cbpwm::OneShotMode::kNone) {
      response->success = false;
      response->message = "Unsupported mode: " + request->mode;
      return;
    }

    OneShotRequest run_request;
    run_request.mode = run_mode;
    run_request.target_block_id = request->target_block_id;
    run_request.enable_debug = request->enable_debug;
    run_request.registration_timeout_s =
      request->timeout_s > 0.0f ? static_cast<double>(request->timeout_s) : 3.0;

    if (run_mode == cbpwm::OneShotMode::kRefineGrasped && run_request.target_block_id.empty()) {
      run_request.target_block_id = resolveGraspedBlockId();
      if (run_request.target_block_id.empty()) {
        response->success = false;
        response->message = "No grasped block found (TASK_MOVE) and no target_block_id provided.";
        return;
      }
    }

    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      if (active_one_shot_.mode != cbpwm::OneShotMode::kNone) {
        response->success = false;
        response->message = "Another one-shot request is already running.";
        return;
      }
      run_request.sequence = ++one_shot_sequence_counter_;
      active_one_shot_ = run_request;
    }

    {
      std::lock_guard<std::mutex> lock(mode_mutex_);
      debug_detection_overlay_enabled_ = request->enable_debug;
      pipeline_mode_ = cbpwm::PipelineMode::kFull;
    }

    RCLCPP_INFO(
      get_logger(),
      "Scheduled one-shot pose estimation: mode=%s target=%s",
      cbpwm::oneShotModeToString(run_mode),
      run_request.target_block_id.empty() ? "<all>" : run_request.target_block_id.c_str());

    const double timeout_s = request->timeout_s > 0.0f ? request->timeout_s : 5.0;
    {
      std::unique_lock<std::mutex> lock(one_shot_mutex_);
      const bool done = one_shot_cv_.wait_for(
        lock,
        std::chrono::duration<double>(timeout_s),
        [this, &run_request]() {
          return one_shot_done_sequence_ >= run_request.sequence;
        });
      if (!done) {
        if (active_one_shot_.sequence == run_request.sequence) {
          active_one_shot_ = OneShotRequest{};
        }
        (void)applyPerceptionMode("IDLE");
        response->success = false;
        response->message = "Timed out waiting for one-shot result.";
        response->blocks = latestWorldSnapshot();
        return;
      }
      response->success = one_shot_last_success_;
      response->message = one_shot_last_message_;
    }

    response->blocks = latestWorldSnapshot();
  }

  void handleSetMode(
    const std::shared_ptr<SetModeSrv::Request> request,
    std::shared_ptr<SetModeSrv::Response> response)
  {
    if (!applyPerceptionMode(request->mode)) {
      response->success = false;
      response->message = "Unsupported mode: " + request->mode;
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mode_mutex_);
      debug_detection_overlay_enabled_ = request->enable_debug;
    }

    response->success = true;
    response->message = "Mode applied: " + request->mode;
  }

  void handleGetCoarseBlocks(
    const std::shared_ptr<GetCoarseSrv::Request> request,
    std::shared_ptr<GetCoarseSrv::Response> response)
  {
    (void)request;
    response->success = true;
    response->blocks = latestWorldSnapshot();
    response->message = "ok";
    RCLCPP_INFO(
      get_logger(),
      "GetCoarseBlocks -> %zu blocks (stamp=%u.%u)",
      response->blocks.blocks.size(),
      response->blocks.header.stamp.sec,
      response->blocks.header.stamp.nanosec);
  }

  static bool isKnownTaskStatus(int32_t task_status)
  {
    return task_status == Block::TASK_UNKNOWN ||
           task_status == Block::TASK_FREE ||
           task_status == Block::TASK_MOVE ||
           task_status == Block::TASK_PLACED ||
           task_status == Block::TASK_REMOVED;
  }

  void handleSetBlockTaskStatus(
    const std::shared_ptr<SetBlockTaskStatusSrv::Request> request,
    std::shared_ptr<SetBlockTaskStatusSrv::Response> response)
  {
    const std::string block_id = request->block_id;
    const int32_t target_task_status = request->task_status;

    if (block_id.empty()) {
      response->success = false;
      response->message = "block_id must not be empty.";
      RCLCPP_WARN(get_logger(), "SetBlockTaskStatus rejected: %s", response->message.c_str());
      return;
    }
    if (!isKnownTaskStatus(target_task_status)) {
      response->success = false;
      response->message = "Unsupported task_status: " + std::to_string(target_task_status);
      RCLCPP_WARN(get_logger(), "SetBlockTaskStatus rejected: %s", response->message.c_str());
      return;
    }

    std_msgs::msg::Header publish_header;
    publish_header.stamp = now();
    {
      const auto snapshot = latestWorldSnapshot();
      publish_header.frame_id = snapshot.header.frame_id.empty() ? world_frame_ : snapshot.header.frame_id;
    }

    int32_t prev_task_status = Block::TASK_UNKNOWN;
    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      const auto it = persistent_world_.find(block_id);
      if (it == persistent_world_.end()) {
        response->success = false;
        response->message = "Unknown block_id: " + block_id;
        RCLCPP_WARN(get_logger(), "SetBlockTaskStatus rejected: %s", response->message.c_str());
        return;
      }

      prev_task_status = it->second.task_status;
      if (!cbpwm::isValidTaskTransition(prev_task_status, target_task_status)) {
        response->success = false;
        response->message =
          std::string("Invalid task transition: ") +
          cbpwm::taskStatusToString(prev_task_status) + " -> " +
          cbpwm::taskStatusToString(target_task_status);
        RCLCPP_WARN(
          get_logger(),
          "SetBlockTaskStatus rejected for block '%s': %s",
          block_id.c_str(),
          response->message.c_str());
        return;
      }

      it->second.task_status = target_task_status;
      // Keep block alive and reflect semantic state change immediately.
      it->second.last_seen = publish_header.stamp;
    }

    publishPersistentWorld(publish_header);

    response->success = true;
    response->message =
      std::string("Updated block '") + block_id + "' task_status to " +
      cbpwm::taskStatusToString(target_task_status);
    RCLCPP_INFO(
      get_logger(),
      "SetBlockTaskStatus applied: block '%s' %s -> %s",
      block_id.c_str(),
      cbpwm::taskStatusToString(prev_task_status),
      cbpwm::taskStatusToString(target_task_status));
  }

  void publishWorldMarkers(const std_msgs::msg::Header & header, const std::vector<Block> & blocks)
  {
    auto marker_header = header;
    marker_header.stamp = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    const auto markers = cbpwm::buildWorldMarkers(marker_header, blocks, world_frame_);
    marker_pub_->publish(markers);

    if (!blocks.empty()) {
      const auto & b0 = blocks.front();
      RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Published markers: %zu blocks in frame '%s' (first: id=%s pos=[%.3f, %.3f, %.3f])",
        blocks.size(),
        world_frame_.c_str(),
        b0.id.c_str(),
        b0.pose.position.x,
        b0.pose.position.y,
        b0.pose.position.z);
    }
  }

  void publishPersistentWorld(const std_msgs::msg::Header & header)
  {
    BlockArray out;
    out.header = header;

    const rclcpp::Time now_stamp(header.stamp);
    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      for (auto it = persistent_world_.begin(); it != persistent_world_.end();) {
        const rclcpp::Time seen(it->second.last_seen);
        if ((now_stamp - seen).seconds() > object_timeout_s_) {
          it = persistent_world_.erase(it);
          continue;
        }
        out.blocks.push_back(it->second);
        ++it;
      }
    }

    world_pub_->publish(out);
    updateLatestWorldCache(out);
    publishWorldMarkers(out.header, out.blocks);
  }

  void publishDetectionOverlay(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const vision_msgs::msg::Detection2DArray & detections,
    const sensor_msgs::msg::Image & mask_msg)
  {
    cv::Mat img = toCvBgr(*image);

    if (!mask_msg.data.empty()) {
      cv::Mat mask = toCvMono(mask_msg);
      overlayMask(img, mask, cv::Scalar(255, 255, 0), 0.35);
    }

    drawDetectionBoxes(img, detections, cv::Scalar(0, 255, 0));
    auto out = cv_bridge::CvImage(image->header, "bgr8", img).toImageMsg();
    det_debug_pub_->publish(*out);
  }

  void recordTiming(int64_t seg_ms, int64_t track_ms, int64_t reg_ms, int64_t total_ms)
  {
    if (!perf_log_timing_enabled_) {
      return;
    }

    std::lock_guard<std::mutex> lock(perf_mutex_);
    perf_timing_count_++;
    perf_seg_sum_ms_ += seg_ms;
    perf_track_sum_ms_ += track_ms;
    perf_reg_sum_ms_ += reg_ms;
    perf_total_sum_ms_ += total_ms;

    const uint64_t n = perf_timing_count_;
    if ((n % static_cast<uint64_t>(perf_log_every_n_frames_)) != 0U) {
      return;
    }

    const double avg_seg = static_cast<double>(perf_seg_sum_ms_) / static_cast<double>(n);
    const double avg_track = static_cast<double>(perf_track_sum_ms_) / static_cast<double>(n);
    const double avg_reg = static_cast<double>(perf_reg_sum_ms_) / static_cast<double>(n);
    const double avg_total = static_cast<double>(perf_total_sum_ms_) / static_cast<double>(n);

    RCLCPP_INFO(
      get_logger(),
      "Timing avg over %llu frames | seg %.1f ms | track %.1f ms | reg %.1f ms | total %.1f ms | dropped busy %llu | dropped sync %llu",
      static_cast<unsigned long long>(n),
      avg_seg,
      avg_track,
      avg_reg,
      avg_total,
      static_cast<unsigned long long>(dropped_busy_frames_.load()),
      static_cast<unsigned long long>(dropped_sync_frames_.load()));
  }

  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
  {
    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      if (active_one_shot_.mode == cbpwm::OneShotMode::kNone) {
        return;
      }
    }

    bool expected = false;
    if (!busy_.compare_exchange_strong(expected, true)) {
      dropped_busy_frames_.fetch_add(1);
      return;
    }

    const rclcpp::Time t_img(image->header.stamp);
    const rclcpp::Time t_cloud(cloud->header.stamp);
    const double dt = std::abs((t_img - t_cloud).seconds());

    if (dt > max_sync_delta_s_) {
      dropped_sync_frames_.fetch_add(1);
      RCLCPP_WARN(
        get_logger(),
        "Dropped frame pair due to sync delta %.4f s (> %.4f s)",
        dt,
        max_sync_delta_s_);
      resetBusy();
      return;
    }

    processFrame(image, cloud);
  }

  void handleOneShotSegmentationResponse(
    rclcpp::Client<SegmentSrv>::SharedFuture seg_future,
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    std::chrono::steady_clock::time_point t_start,
    const OneShotRequest & run_request)
  {
    try {
      auto seg_res = seg_future.get();
      const auto t_after_seg = std::chrono::steady_clock::now();
      const size_t seg_detection_count =
        seg_res ? seg_res->detections.detections.size() : 0U;

      if (!seg_res || !seg_res->success) {
        publishPersistentWorld(cloud->header);
        completeOneShotRequest(run_request.sequence, false, "Segmentation failed.");
        resetBusy();
        return;
      }

      RCLCPP_INFO(
        get_logger(),
        "One-shot %s: segmentation detections=%zu",
        cbpwm::oneShotModeToString(run_request.mode),
        seg_detection_count);

      if (debug_detection_overlay_enabled_ && det_debug_pub_) {
        publishDetectionOverlay(image, seg_res->detections, seg_res->mask);
      }

      const auto t_after_track = t_after_seg;
      const size_t tracked_count = seg_detection_count;
      RCLCPP_INFO(
        get_logger(),
        "One-shot %s: detections=%zu tracked=%zu (tracker bypassed)",
        cbpwm::oneShotModeToString(run_request.mode),
        seg_detection_count,
        tracked_count);

      if (seg_res->detections.detections.empty()) {
        const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          t_after_seg - t_start).count();
        recordTiming(total_ms, 0, 0, total_ms);
        publishPersistentWorld(cloud->header);

        if (run_request.mode == cbpwm::OneShotMode::kSceneDiscovery) {
          completeOneShotRequest(
            run_request.sequence,
            true,
            "Scene discovery finished (detections=0, tracked=0, registrations=0).");
        } else {
          completeOneShotRequest(
            run_request.sequence,
            false,
            "Requested block was not detected.");
        }
        resetBusy();
        return;
      }

      auto candidates = cbpwm::buildRegistrationCandidates(
        *seg_res, run_request.mode, run_request.target_block_id);

      const size_t registration_candidates = candidates.size();
      RCLCPP_INFO(
        get_logger(),
        "One-shot %s: detections=%zu tracked=%zu registration_candidates=%zu",
        cbpwm::oneShotModeToString(run_request.mode),
        seg_detection_count,
        tracked_count,
        registration_candidates);

      if (registration_candidates == 0) {
        const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          t_after_track - t_start).count();
        recordTiming(total_ms, 0, 0, total_ms);
        publishPersistentWorld(cloud->header);

        if (run_request.mode == cbpwm::OneShotMode::kSceneDiscovery) {
          completeOneShotRequest(
            run_request.sequence,
            true,
            "Scene discovery finished (detections=" + std::to_string(seg_detection_count) +
            ", tracked=" + std::to_string(tracked_count) +
            ", registration_candidates=0, registrations=0).");
        } else {
          completeOneShotRequest(
            run_request.sequence,
            false,
            "Requested block was not available for registration "
            "(detections=" + std::to_string(seg_detection_count) +
            ", tracked=" + std::to_string(tracked_count) + ").");
        }

        resetBusy();
        return;
      }

      if (!action_client_->action_server_is_ready()) {
        RCLCPP_WARN(get_logger(), "Registration action unavailable.");
        completeOneShotRequest(run_request.sequence, false, "Registration action unavailable.");
        resetBusy();
        return;
      }

      const auto t_reg_start = std::chrono::steady_clock::now();
      cbpwm::RegistrationCounters counters;
      Eigen::Vector3d camera_origin_world = Eigen::Vector3d::Zero();
      std::string camera_origin_reason;
      const bool have_camera_origin = lookupFrameOriginInWorld(
        cloud->header, camera_origin_world, camera_origin_reason);
      if (!have_camera_origin) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Scene-discovery coarse center offset disabled for this frame: %s",
          camera_origin_reason.c_str());
      }
      cbpwm::SceneFlowRequest scene_req;
      scene_req.mode = run_request.mode;
      scene_req.target_block_id = run_request.target_block_id;
      scene_req.registration_timeout_s = run_request.registration_timeout_s;
      scene_req.refine_target_max_distance_m = refine_target_max_distance_m_;

      cbpwm::SceneFlowRuntime scene_rt;
      scene_rt.logger = get_logger();
      scene_rt.run_registration =
        [this](
        uint32_t detection_id,
        const sensor_msgs::msg::Image & mask,
        const sensor_msgs::msg::PointCloud2 & in_cloud,
        const std_msgs::msg::Header & header,
        double timeout_s,
        Block & out_block,
        std::string & out_reason) {
          return runRegistrationSync(
            detection_id,
            mask,
            in_cloud,
            header,
            timeout_s,
            out_block,
            out_reason);
        };
      scene_rt.upsert_block =
        [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
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
      scene_rt.get_expected_target =
        [this](const std::string & target_id, Block & out_target) {
          std::lock_guard<std::mutex> lock(persistent_world_mutex_);
          const auto it = persistent_world_.find(target_id);
          if (it == persistent_world_.end()) {
            return false;
          }
          out_target = it->second;
          return true;
        };
      scene_rt.try_coarse_fallback =
        [this, have_camera_origin, &camera_origin_world](
        uint32_t detection_id,
        const std::string & registration_reason,
        const sensor_msgs::msg::Image & mask,
        const sensor_msgs::msg::PointCloud2 & in_cloud,
        const cbpwm::SceneFlowRequest & request,
        cbpwm::RegistrationCounters & out_counters) {
          return trySceneDiscoveryCoarseFallback(
            detection_id,
            registration_reason,
            mask,
            in_cloud,
            request,
            have_camera_origin ? &camera_origin_world : nullptr,
            out_counters);
        };

      cbpwm::processRegistrationCandidates(
        candidates,
        *cloud,
        scene_req,
        scene_rt,
        counters);

      const auto t_reg_end = std::chrono::steady_clock::now();
      const auto seg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_after_seg - t_start).count();
      const auto track_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_after_track - t_after_seg).count();
      const auto reg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_reg_end - t_reg_start).count();
      const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_reg_end - t_start).count();
      recordTiming(seg_ms, track_ms, reg_ms, total_ms);

      publishPersistentWorld(cloud->header);
      completeOneShotRequest(
        run_request.sequence,
        counters.registrations_ok > 0 || run_request.mode == cbpwm::OneShotMode::kSceneDiscovery,
        "Pose estimation finished (detections=" + std::to_string(seg_detection_count) +
        ", tracked=" + std::to_string(tracked_count) +
        ", registration_candidates=" + std::to_string(registration_candidates) +
        ", registrations=" + std::to_string(counters.registrations_ok) +
        ", coarse_registrations=" + std::to_string(counters.coarse_upserts_ok) + ").");

      resetBusy();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Segmentation stage failed: %s", e.what());
      completeOneShotRequest(run_request.sequence, false, "Segmentation stage exception.");
      resetBusy();
    }
  }

  void processFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
  {
    const auto t_start = std::chrono::steady_clock::now();

    OneShotRequest run_request;
    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      run_request = active_one_shot_;
    }

    if (run_request.mode == cbpwm::OneShotMode::kNone ||
      pipeline_mode_ == cbpwm::PipelineMode::kIdle)
    {
      resetBusy();
      return;
    }

    if (run_request.mode == cbpwm::OneShotMode::kRefineGrasped && refine_grasped_use_fk_roi_) {
      processRefineGraspedWithFkRoi(image, cloud, run_request, t_start);
      return;
    }
    if (run_request.mode == cbpwm::OneShotMode::kRefineBlock &&
      tryProcessRefineBlockWithPoseRoi(image, cloud, run_request, t_start))
    {
      return;
    }

    if (!segment_client_->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "Segmentation service unavailable.");
      resetBusy();
      return;
    }

    auto seg_req = std::make_shared<SegmentSrv::Request>();
    seg_req->image = *image;
    seg_req->return_debug = false;

    segment_client_->async_send_request(
      seg_req,
      [this, image, cloud, t_start, run_request](rclcpp::Client<SegmentSrv>::SharedFuture seg_future) {
        handleOneShotSegmentationResponse(seg_future, image, cloud, t_start, run_request);
      });
  }

