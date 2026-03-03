#include <algorithm>
#include <cctype>
#include <utility>

#include "concrete_block_perception/utils/world_model_utils.hpp"

namespace cbp::world_model
{

using concrete_block_perception::msg::Block;

std::string normalizeMode(std::string mode)
{
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return mode;
}

PipelineMode parsePipelineMode(const std::string & mode)
{
  if (mode == "idle") {
    return PipelineMode::kIdle;
  }
  if (mode == "segment") {
    return PipelineMode::kSegment;
  }
  if (mode == "track") {
    return PipelineMode::kTrack;
  }
  if (mode == "register") {
    return PipelineMode::kRegister;
  }
  return PipelineMode::kFull;
}

const char * pipelineModeToString(PipelineMode mode)
{
  switch (mode) {
    case PipelineMode::kIdle:
      return "idle";
    case PipelineMode::kSegment:
      return "segment";
    case PipelineMode::kTrack:
      return "track";
    case PipelineMode::kRegister:
      return "register";
    case PipelineMode::kFull:
    default:
      return "full";
  }
}

bool resolvePerceptionModeConfig(const std::string & mode, PerceptionModeConfig & out)
{
  const std::string m = normalizeMode(mode);

  if (m == "IDLE") {
    out = {PerceptionMode::kIdle, PipelineMode::kIdle, false, "Perception mode set: IDLE (no processing)"};
    return true;
  }
  if (m == "SCENE_SCAN") {
    out = {PerceptionMode::kSceneScan, PipelineMode::kTrack, false, "Perception mode set: SCENE_SCAN (track + coarse world publish)"};
    return true;
  }
  if (m == "PRE_GRASP") {
    out = {PerceptionMode::kPreGrasp, PipelineMode::kFull, true, "Perception mode set: PRE_GRASP (registration on demand)"};
    return true;
  }
  if (m == "GRASP_EXECUTE") {
    out = {PerceptionMode::kGraspExecute, PipelineMode::kTrack, false, "Perception mode set: GRASP_EXECUTE (track only)"};
    return true;
  }
  if (m == "TRANSPORT") {
    out = {PerceptionMode::kTransport, PipelineMode::kSegment, false, "Perception mode set: TRANSPORT (segment only)"};
    return true;
  }
  if (m == "PRE_ASSEMBLY") {
    out = {PerceptionMode::kPreAssembly, PipelineMode::kFull, true, "Perception mode set: PRE_ASSEMBLY (registration on demand)"};
    return true;
  }
  if (m == "ASSEMBLY_EXECUTE") {
    out = {PerceptionMode::kAssemblyExecute, PipelineMode::kTrack, false, "Perception mode set: ASSEMBLY_EXECUTE (track only)"};
    return true;
  }

  if (m == "SEGMENT") {
    out = {PerceptionMode::kSceneScan, PipelineMode::kSegment, false, "Perception mode set: SEGMENT"};
    return true;
  }
  if (m == "TRACK") {
    out = {PerceptionMode::kSceneScan, PipelineMode::kTrack, false, "Perception mode set: TRACK"};
    return true;
  }
  if (m == "REGISTER") {
    out = {PerceptionMode::kSceneScan, PipelineMode::kRegister, false, "Perception mode set: REGISTER"};
    return true;
  }
  if (m == "FULL") {
    out = {PerceptionMode::kSceneScan, PipelineMode::kFull, false, "Perception mode set: FULL"};
    return true;
  }

  return false;
}

const char * perceptionModeToString(PerceptionMode mode)
{
  switch (mode) {
    case PerceptionMode::kIdle:
      return "IDLE";
    case PerceptionMode::kSceneScan:
      return "SCENE_SCAN";
    case PerceptionMode::kPreGrasp:
      return "PRE_GRASP";
    case PerceptionMode::kGraspExecute:
      return "GRASP_EXECUTE";
    case PerceptionMode::kTransport:
      return "TRANSPORT";
    case PerceptionMode::kPreAssembly:
      return "PRE_ASSEMBLY";
    case PerceptionMode::kAssemblyExecute:
    default:
      return "ASSEMBLY_EXECUTE";
  }
}

OneShotMode parseOneShotMode(const std::string & mode)
{
  const std::string m = normalizeMode(mode);
  if (m == "SCENE_DISCOVERY") {
    return OneShotMode::kSceneDiscovery;
  }
  if (m == "REFINE_BLOCK") {
    return OneShotMode::kRefineBlock;
  }
  if (m == "REFINE_GRASPED") {
    return OneShotMode::kRefineGrasped;
  }
  return OneShotMode::kNone;
}

const char * oneShotModeToString(OneShotMode mode)
{
  switch (mode) {
    case OneShotMode::kSceneDiscovery:
      return "SCENE_DISCOVERY";
    case OneShotMode::kRefineBlock:
      return "REFINE_BLOCK";
    case OneShotMode::kRefineGrasped:
      return "REFINE_GRASPED";
    case OneShotMode::kNone:
    default:
      return "NONE";
  }
}

visualization_msgs::msg::MarkerArray buildWorldMarkers(
  const std_msgs::msg::Header & header,
  const std::vector<Block> & blocks,
  const std::string & world_frame)
{
  constexpr double kMarkerWidthM = 0.9;
  constexpr double kMarkerHeightM = 0.6;
  constexpr double kMarkerDepthM = 0.6;

  visualization_msgs::msg::MarkerArray ma;
  auto marker_header = header;
  marker_header.frame_id = world_frame;

  int marker_id = 0;
  for (const auto & b : blocks) {
    visualization_msgs::msg::Marker m;
    m.header = marker_header;
    m.ns = "cbp_blocks";
    m.id = marker_id++;
    m.type = (b.pose_status == Block::POSE_COARSE) ?
      visualization_msgs::msg::Marker::SPHERE :
      visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose = b.pose;
    m.scale.x = kMarkerWidthM;
    m.scale.y = kMarkerHeightM;
    m.scale.z = kMarkerDepthM;

    if (b.task_status == Block::TASK_REMOVED) {
      m.color.r = 0.9f;
      m.color.g = 0.1f;
      m.color.b = 0.1f;
    } else if (b.task_status == Block::TASK_PLACED) {
      m.color.r = 0.1f;
      m.color.g = 0.9f;
      m.color.b = 0.9f;
    } else if (b.task_status == Block::TASK_MOVE) {
      m.color.r = 0.2f;
      m.color.g = 0.4f;
      m.color.b = 1.0f;
    } else if (b.pose_status == Block::POSE_PRECISE) {
      m.color.r = 0.1f;
      m.color.g = 0.8f;
      m.color.b = 0.2f;
    } else if (b.pose_status == Block::POSE_COARSE) {
      m.color.r = 1.0f;
      m.color.g = 0.8f;
      m.color.b = 0.1f;
    } else {
      m.color.r = 0.5f;
      m.color.g = 0.5f;
      m.color.b = 0.5f;
    }
    m.color.a = 0.6f;
    ma.markers.push_back(std::move(m));

    visualization_msgs::msg::Marker label;
    label.header = marker_header;
    label.ns = "cbp_block_ids";
    label.id = marker_id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose = b.pose;
    label.pose.position.z += 0.7;
    label.scale.z = 0.2;
    label.color.r = 1.0f;
    label.color.g = 1.0f;
    label.color.b = 1.0f;
    label.color.a = 0.95f;
    label.text = b.id;
    ma.markers.push_back(std::move(label));
  }

  return ma;
}

}  // namespace cbp::world_model
