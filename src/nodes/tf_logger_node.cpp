#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <fstream>

class TfLogger : public rclcpp::Node
{
public:
  TfLogger()
  : Node("tf_logger"),
    buffer_(this->get_clock()),
    listener_(buffer_)
  {
    file_.open("tool_contact_point.csv");
    file_ << "time,x,y,z,qx,qy,qz,qw\n";

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(20),
      std::bind(&TfLogger::tick, this));
  }

private:
  void tick()
  {
    try {
      auto tf = buffer_.lookupTransform(
        "base_link",
        "tool_contact_point",
        tf2::TimePointZero);

      double t =
        tf.header.stamp.sec +
        1e-9 * tf.header.stamp.nanosec;

      file_
        << t << ","
        << tf.transform.translation.x << ","
        << tf.transform.translation.y << ","
        << tf.transform.translation.z << ","
        << tf.transform.rotation.x << ","
        << tf.transform.rotation.y << ","
        << tf.transform.rotation.z << ","
        << tf.transform.rotation.w << "\n";
    } catch (tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    }
  }

  tf2_ros::Buffer buffer_;
  tf2_ros::TransformListener listener_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::ofstream file_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TfLogger>());
  rclcpp::shutdown();
}
