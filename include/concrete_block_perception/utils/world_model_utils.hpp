#pragma once

#include <string>
#include <vector>

#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "concrete_block_perception/msg/block.hpp"

namespace cbp::world_model
{

enum class PipelineMode
{
  kIdle,
  kSegment,
  kTrack,
  kRegister,
  kFull
};

enum class PerceptionMode
{
  kIdle,
  kSceneScan,
  kPreGrasp,
  kGraspExecute,
  kTransport,
  kPreAssembly,
  kAssemblyExecute
};

enum class OneShotMode
{
  kNone,
  kSceneDiscovery,
  kRefineBlock,
  kRefineGrasped
};

struct PerceptionModeConfig
{
  PerceptionMode perception_mode{PerceptionMode::kIdle};
  PipelineMode pipeline_mode{PipelineMode::kIdle};
  bool registration_on_demand{false};
  const char * log_message{"Perception mode set"};
};

std::string normalizeMode(std::string mode);

PipelineMode parsePipelineMode(const std::string & mode);
const char * pipelineModeToString(PipelineMode mode);

bool resolvePerceptionModeConfig(const std::string & mode, PerceptionModeConfig & out);
const char * perceptionModeToString(PerceptionMode mode);

OneShotMode parseOneShotMode(const std::string & mode);
const char * oneShotModeToString(OneShotMode mode);

visualization_msgs::msg::MarkerArray buildWorldMarkers(
  const std_msgs::msg::Header & header,
  const std::vector<concrete_block_perception::msg::Block> & blocks,
  const std::string & world_frame);

}  // namespace cbp::world_model
