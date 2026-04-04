#ifndef PANDA_TRY_RVIZ__PANDA_TRY_CONTROL_PANEL_HPP_
#define PANDA_TRY_RVIZ__PANDA_TRY_CONTROL_PANEL_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>

#include <QLabel>
#include <QPushButton>
#include <QSlider>

namespace panda_try_rviz
{

class PandaTryControlPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit PandaTryControlPanel(QWidget * parent = nullptr);
  ~PandaTryControlPanel() override;

  void onInitialize() override;

private:
  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_ptr_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr save_point_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr collision_sub_;

  QLabel * title_label_{};
  QLabel * collision_label_{};
  QPushButton * save_button_{};

  std::array<QSlider *, 7> sliders_{};
  std::array<QLabel *, 7> value_labels_{};

  std::array<double, 7> joint_positions_rad_{};

  static constexpr const char * kJointNames[7] = {
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"
  };

  static constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
  static constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

  static double clampDeg(double v_deg, int idx);
  static double jointMinDeg(int idx);
  static double jointMaxDeg(int idx);

  void onJointSliderChanged(int idx, int slider_value);
  void onSavePointClicked();
  void onCollisionStatus(const std_msgs::msg::Bool & msg);

  void publishCurrentJointState();
  void refreshCollisionLabel(bool in_collision);
  void refreshJointValueLabel(int idx);
};

}  // namespace panda_try_rviz

#endif  // PANDA_TRY_RVIZ__PANDA_TRY_CONTROL_PANEL_HPP_
