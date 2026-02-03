# Short-lived image-space detection with persistent detection_id

uint32 detection_id

# Original detection (bbox, class, confidence)
vision_msgs/Detection2D detection

# Segmentation mask corresponding to this detection
sensor_msgs/Image mask

# Number of consecutive frames this detection has been observed
uint32 age

# Number of consecutive frames this detection was missing
uint32 misses

# Timestamp of last update
builtin_interfaces/Time stamp
