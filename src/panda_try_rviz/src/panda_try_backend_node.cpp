#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <moveit_msgs/msg/contact_information.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace panda_try_rviz
{

class PandaTryBackendNode : public rclcpp::Node
{
public:
  PandaTryBackendNode()
  : Node("panda_try_backend_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    group_name_ = this->declare_parameter<std::string>("group_name", "panda_arm");
    marker_frame_ = this->declare_parameter<std::string>("marker_frame", "world");
    ee_link_frame_ = this->declare_parameter<std::string>("ee_link_frame", "panda_try_/panda_hand");

    command_joint_topic_ = this->declare_parameter<std::string>(
      "command_joint_topic", "/panda_try/command_joint_states");
    out_joint_topic_ = this->declare_parameter<std::string>(
      "out_joint_topic", "/panda_try/joint_states");
    save_point_topic_ = this->declare_parameter<std::string>(
      "save_point_topic", "/panda_try/save_ee_point");
    collision_topic_ = this->declare_parameter<std::string>(
      "collision_topic", "/panda_try/self_collision");
    marker_topic_ = this->declare_parameter<std::string>(
      "marker_topic", "/panda_try/markers");
    state_validity_service_ = this->declare_parameter<std::string>(
      "state_validity_service", "/check_state_validity");

    const auto initial_q = this->declare_parameter<std::vector<double>>(
      "initial_joint_positions",
      std::vector<double>{
        -0.01779206, -0.76012354, 0.01978261, -2.34205014,
        0.02984053, 1.54119353, 0.75344866
      });

    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(out_joint_topic_, 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);
    collision_pub_ = this->create_publisher<std_msgs::msg::Bool>(collision_topic_, 10);

    command_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      command_joint_topic_, 10,
      std::bind(&PandaTryBackendNode::onCommandJointState, this, std::placeholders::_1));

    save_point_sub_ = this->create_subscription<std_msgs::msg::Empty>(
      save_point_topic_, 10,
      std::bind(&PandaTryBackendNode::onSavePoint, this, std::placeholders::_1));

    state_validity_client_ = this->create_client<moveit_msgs::srv::GetStateValidity>(state_validity_service_);

    check_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&PandaTryBackendNode::onCheckTimer, this));

    marker_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(300),
      std::bind(&PandaTryBackendNode::publishMarkers, this));

    collision_msg_.data = false;

    current_joint_state_.name = defaultJointNames();
    current_joint_state_.position.resize(7, 0.0);
    for (size_t i = 0; i < std::min<size_t>(7, initial_q.size()); ++i) {
      current_joint_state_.position[i] = initial_q[i];
    }
    current_joint_state_.header.stamp = this->now();
    has_joint_state_ = true;
    joint_state_pub_->publish(current_joint_state_);
    publishCollisionStatus(false);
    publishMarkers();

    RCLCPP_INFO(
      this->get_logger(),
      "panda_try backend started. command_topic=%s, out_topic=%s, marker_topic=%s, service=%s",
      command_joint_topic_.c_str(), out_joint_topic_.c_str(), marker_topic_.c_str(), state_validity_service_.c_str());
  }

private:
  static std::vector<std::string> defaultJointNames()
  {
    return {
      "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
      "panda_joint5", "panda_joint6", "panda_joint7"
    };
  }

  static bool isPandaLinkName(const std::string & name)
  {
    if (name.empty()) {
      return false;
    }
    return name.find("panda_") != std::string::npos;
  }

  static bool pointFinite(const geometry_msgs::msg::Point & p)
  {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
  }

  static bool almostSamePoint(
    const geometry_msgs::msg::Point & a,
    const geometry_msgs::msg::Point & b,
    double eps = 1e-5)
  {
    return std::abs(a.x - b.x) <= eps && std::abs(a.y - b.y) <= eps && std::abs(a.z - b.z) <= eps;
  }

  void onCommandJointState(const sensor_msgs::msg::JointState & msg)
  {
    if (msg.position.size() < 7) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1500,
        "Received command_joint_states with less than 7 positions. Ignored.");
      return;
    }

    current_joint_state_.header.stamp = this->now();
    current_joint_state_.name = msg.name;
    current_joint_state_.position.assign(msg.position.begin(), msg.position.begin() + 7);

    if (current_joint_state_.name.size() < 7) {
      current_joint_state_.name = defaultJointNames();
    } else if (current_joint_state_.name.size() > 7) {
      current_joint_state_.name.resize(7);
    }

    has_joint_state_ = true;
    joint_state_pub_->publish(current_joint_state_);
  }

  void onSavePoint(const std_msgs::msg::Empty &)
  {
    geometry_msgs::msg::Point p;
    if (!lookupCurrentEePoint(p)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "Failed to save EE point: TF lookup %s -> %s failed.",
        marker_frame_.c_str(), ee_link_frame_.c_str());
      return;
    }

    saved_points_.push_back(p);
    RCLCPP_INFO(
      this->get_logger(),
      "Saved EE point #%zu: (%.4f, %.4f, %.4f)",
      saved_points_.size(), p.x, p.y, p.z);
    publishMarkers();
  }

  bool lookupCurrentEePoint(geometry_msgs::msg::Point & out_p)
  {
    try {
      const geometry_msgs::msg::TransformStamped tf = tf_buffer_.lookupTransform(
        marker_frame_, ee_link_frame_, tf2::TimePointZero);
      out_p.x = tf.transform.translation.x;
      out_p.y = tf.transform.translation.y;
      out_p.z = tf.transform.translation.z;
      return pointFinite(out_p);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_DEBUG(this->get_logger(), "TF lookup failed: %s", ex.what());
      return false;
    }
  }

  void onCheckTimer()
  {
    if (!has_joint_state_) {
      return;
    }
    if (request_inflight_) {
      return;
    }
    if (!state_validity_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Service %s is not ready yet.", state_validity_service_.c_str());
      return;
    }

    auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
    req->group_name = group_name_;
    req->robot_state.joint_state = current_joint_state_;

    request_inflight_ = true;
    using SharedFuture = rclcpp::Client<moveit_msgs::srv::GetStateValidity>::SharedFuture;
    state_validity_client_->async_send_request(
      req,
      [this](SharedFuture future) {
        this->onStateValidityResponse(future);
      });
  }

  void onStateValidityResponse(
    const rclcpp::Client<moveit_msgs::srv::GetStateValidity>::SharedFuture & future)
  {
    request_inflight_ = false;

    auto resp = future.get();
    if (!resp) {
      return;
    }

    std::vector<geometry_msgs::msg::Point> contacts;
    contacts.reserve(resp->contacts.size());

    for (const moveit_msgs::msg::ContactInformation & c : resp->contacts) {
      const bool self_pair = isPandaLinkName(c.contact_body_1) && isPandaLinkName(c.contact_body_2);
      if (!self_pair) {
        continue;
      }
      if (!pointFinite(c.position)) {
        continue;
      }

      bool dup = false;
      for (const auto & p : contacts) {
        if (almostSamePoint(p, c.position)) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        contacts.push_back(c.position);
      }
    }

    collision_contacts_ = contacts;

    const bool in_collision = (!resp->valid) || (!collision_contacts_.empty());
    publishCollisionStatus(in_collision);
    publishMarkers();
  }

  void publishCollisionStatus(bool in_collision)
  {
    collision_msg_.data = in_collision;
    collision_pub_->publish(collision_msg_);
  }

  void publishMarkers()
  {
    visualization_msgs::msg::MarkerArray ma;
    const auto stamp = this->now();

    auto clear = visualization_msgs::msg::Marker();
    clear.header.stamp = stamp;
    clear.header.frame_id = marker_frame_;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(clear);

    int id = 1;

    for (const auto & p : saved_points_) {
      auto m = visualization_msgs::msg::Marker();
      m.header.stamp = stamp;
      m.header.frame_id = marker_frame_;
      m.ns = "saved_ee_points";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.orientation.w = 1.0;
      m.pose.position = p;
      m.scale.x = 0.03;
      m.scale.y = 0.03;
      m.scale.z = 0.03;
      m.color.r = 1.0F;
      m.color.g = 1.0F;
      m.color.b = 0.0F;
      m.color.a = 0.95F;
      ma.markers.push_back(m);
    }

    std::vector<geometry_msgs::msg::Point> collision_points = collision_contacts_;
    if (collision_msg_.data && collision_points.empty()) {
      geometry_msgs::msg::Point ee_p;
      if (lookupCurrentEePoint(ee_p)) {
        collision_points.push_back(ee_p);
      }
    }

    for (const auto & p : collision_points) {
      auto m = visualization_msgs::msg::Marker();
      m.header.stamp = stamp;
      m.header.frame_id = marker_frame_;
      m.ns = "self_collision_contacts";
      m.id = 1000 + id++;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.orientation.w = 1.0;
      m.pose.position = p;
      m.scale.x = 0.055;
      m.scale.y = 0.055;
      m.scale.z = 0.055;
      m.color.r = 1.0F;
      m.color.g = 0.12F;
      m.color.b = 0.12F;
      m.color.a = 0.95F;
      ma.markers.push_back(m);
    }

    if (collision_msg_.data) {
      auto text = visualization_msgs::msg::Marker();
      text.header.stamp = stamp;
      text.header.frame_id = marker_frame_;
      text.ns = "self_collision_text";
      text.id = 9000;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.orientation.w = 1.0;
      text.pose.position.x = 0.0;
      text.pose.position.y = 0.0;
      text.pose.position.z = 1.05;
      text.scale.z = 0.07;
      text.color.r = 1.0F;
      text.color.g = 0.2F;
      text.color.b = 0.2F;
      text.color.a = 1.0F;
      text.text = "SELF COLLISION";
      ma.markers.push_back(text);
    }

    marker_pub_->publish(ma);
  }

private:
  std::string group_name_;
  std::string marker_frame_;
  std::string ee_link_frame_;

  std::string command_joint_topic_;
  std::string out_joint_topic_;
  std::string save_point_topic_;
  std::string collision_topic_;
  std::string marker_topic_;
  std::string state_validity_service_;

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr collision_pub_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr command_joint_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr save_point_sub_;

  rclcpp::Client<moveit_msgs::srv::GetStateValidity>::SharedPtr state_validity_client_;

  rclcpp::TimerBase::SharedPtr check_timer_;
  rclcpp::TimerBase::SharedPtr marker_timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  sensor_msgs::msg::JointState current_joint_state_;
  bool has_joint_state_{false};
  bool request_inflight_{false};

  std::vector<geometry_msgs::msg::Point> saved_points_;
  std::vector<geometry_msgs::msg::Point> collision_contacts_;

  std_msgs::msg::Bool collision_msg_;
};

}  // namespace panda_try_rviz

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<panda_try_rviz::PandaTryBackendNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
