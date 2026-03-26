# Shape identifiers
uint8 SHAPE_UNKNOWN = 0
uint8 SHAPE_BOX = 1

# Object provenance
uint8 SOURCE_UNKNOWN = 0
uint8 SOURCE_BLOCK = 1
uint8 SOURCE_STATIC_OBSTACLE = 2

string id
string frame_id
uint8 shape_type
uint8 source_type
geometry_msgs/Pose pose
geometry_msgs/Vector3 dimensions
int32 pose_status
int32 task_status
float32 confidence
