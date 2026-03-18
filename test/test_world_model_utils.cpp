#include <gtest/gtest.h>

#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/utils/world_model_utils.hpp"

namespace cbpwm = cbp::world_model;
using concrete_block_perception::msg::Block;

TEST(WorldModelUtils, TaskTransitionRules) {
  EXPECT_TRUE(cbpwm::isValidTaskTransition(Block::TASK_UNKNOWN, Block::TASK_FREE));
  EXPECT_TRUE(cbpwm::isValidTaskTransition(Block::TASK_FREE, Block::TASK_MOVE));
  EXPECT_TRUE(cbpwm::isValidTaskTransition(Block::TASK_MOVE, Block::TASK_PLACED));
  EXPECT_TRUE(cbpwm::isValidTaskTransition(Block::TASK_PLACED, Block::TASK_MOVE));

  EXPECT_FALSE(cbpwm::isValidTaskTransition(Block::TASK_FREE, Block::TASK_PLACED));
  EXPECT_FALSE(cbpwm::isValidTaskTransition(Block::TASK_REMOVED, Block::TASK_FREE));
}

TEST(WorldModelUtils, AssociationDistanceAndConfidenceGating) {
  EXPECT_TRUE(cbpwm::shouldAssociateByDistance(0.20, 0.45, 0.9, 0.25));
  EXPECT_FALSE(cbpwm::shouldAssociateByDistance(0.60, 0.45, 0.9, 0.25));
  EXPECT_FALSE(cbpwm::shouldAssociateByDistance(0.20, 0.45, 0.1, 0.25));
}

TEST(WorldModelUtils, TaskStatusToString) {
  EXPECT_STREQ(cbpwm::taskStatusToString(Block::TASK_FREE), "TASK_FREE");
  EXPECT_STREQ(cbpwm::taskStatusToString(Block::TASK_MOVE), "TASK_MOVE");
  EXPECT_STREQ(cbpwm::taskStatusToString(Block::TASK_PLACED), "TASK_PLACED");
}

TEST(WorldModelUtils, ResolvePerceptionModeConfigForStagedWorkflow) {
  cbpwm::PerceptionModeConfig cfg;

  ASSERT_TRUE(cbpwm::resolvePerceptionModeConfig("IDLE", cfg));
  EXPECT_EQ(cfg.perception_mode, cbpwm::PerceptionMode::kIdle);
  EXPECT_EQ(cfg.pipeline_mode, cbpwm::PipelineMode::kIdle);
  EXPECT_FALSE(cfg.registration_on_demand);

  ASSERT_TRUE(cbpwm::resolvePerceptionModeConfig("PRE_GRASP", cfg));
  EXPECT_EQ(cfg.perception_mode, cbpwm::PerceptionMode::kPreGrasp);
  EXPECT_EQ(cfg.pipeline_mode, cbpwm::PipelineMode::kFull);
  EXPECT_TRUE(cfg.registration_on_demand);

  ASSERT_TRUE(cbpwm::resolvePerceptionModeConfig("ASSEMBLY_EXECUTE", cfg));
  EXPECT_EQ(cfg.perception_mode, cbpwm::PerceptionMode::kAssemblyExecute);
  EXPECT_EQ(cfg.pipeline_mode, cbpwm::PipelineMode::kTrack);
  EXPECT_FALSE(cfg.registration_on_demand);
}

TEST(WorldModelUtils, ParseOneShotModes) {
  EXPECT_EQ(
    cbpwm::parseOneShotMode("SCENE_DISCOVERY"),
    cbpwm::OneShotMode::kSceneDiscovery);
  EXPECT_EQ(
    cbpwm::parseOneShotMode("refine_block"),
    cbpwm::OneShotMode::kRefineBlock);
  EXPECT_EQ(
    cbpwm::parseOneShotMode("Refine_Grasped"),
    cbpwm::OneShotMode::kRefineGrasped);
  EXPECT_EQ(
    cbpwm::parseOneShotMode("unsupported"),
    cbpwm::OneShotMode::kNone);
}
