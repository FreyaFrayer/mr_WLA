#ifndef PANDA_IK_BENCHMARK_VIZ_RVIZ_PLUGINS__IK_BENCHMARK_PLAYBACK_PANEL_HPP_
#define PANDA_IK_BENCHMARK_VIZ_RVIZ_PLUGINS__IK_BENCHMARK_PLAYBACK_PANEL_HPP_

#include <memory>
#include <string>
#include <vector>
#include <map>

#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>

#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QTableWidget>

namespace panda_ik_benchmark_viz_rviz_plugins
{

struct WaypointTick
{
  QString name;
  float progress;  // 0..1 in the shared cycle
  bool is_global;  // true => global, false => greedy
};

class IKBenchmarkTimeline : public QSlider
{
  Q_OBJECT
public:
  explicit IKBenchmarkTimeline(QWidget * parent = nullptr);

  void setTicks(const std::vector<WaypointTick> & ticks);

Q_SIGNALS:
  void tickClicked(float progress);

protected:
  void paintEvent(QPaintEvent * ev) override;
  void mousePressEvent(QMouseEvent * ev) override;

private:
  std::vector<WaypointTick> ticks_;
  int findNearestTickPx(int x_px, int tolerance_px) const;
  int tickXpx(int tick_index) const;
};

class IKBenchmarkPlaybackPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit IKBenchmarkPlaybackPanel(QWidget * parent = nullptr);
  ~IKBenchmarkPlaybackPanel() override;

  void onInitialize() override;

private Q_SLOTS:
  void onTogglePause();
  void onSliderPressed();
  void onSliderReleased();
  void onSliderValueChanged(int value);
  void onTickClicked(float progress);

private:
  // ROS integration
  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_ptr_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pause_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr seek_pub_;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr paused_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr progress_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr time_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr cycle_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr waypoints_sub_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr greedy_js_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr global_js_sub_;

  // UI widgets
  QLabel * status_label_{};
  QPushButton * pause_button_{};
  IKBenchmarkTimeline * slider_{};
  QTableWidget * joint_table_{};

  // State
  bool paused_{false};
  bool dragging_{false};
  float cycle_s_{0.0F};
  float time_s_{0.0F};
  float progress_{0.0F};

  // Waypoint ticks
  std::vector<WaypointTick> ticks_;

  // Joint states
  std::vector<std::string> joint_names_order_;
  std::map<std::string, double> greedy_joint_pos_;
  std::map<std::string, double> global_joint_pos_;

  // Topics (fixed defaults)
  std::string pause_topic_{"/ik_benchmark/playback/pause"};
  std::string seek_topic_{"/ik_benchmark/playback/seek"};
  std::string progress_topic_{"/ik_benchmark/playback/progress"};
  std::string time_topic_{"/ik_benchmark/playback/time_s"};
  std::string cycle_topic_{"/ik_benchmark/playback/cycle_s"};
  std::string paused_topic_{"/ik_benchmark/playback/paused"};
  std::string waypoints_topic_{"/ik_benchmark/playback/waypoints_json"};

  std::string greedy_joint_states_topic_{"/greedy/joint_states"};
  std::string global_joint_states_topic_{"/global/joint_states"};

  void updateUi();
  void updateJointTable();

  void pausedCallback(const std_msgs::msg::Bool & msg);
  void progressCallback(const std_msgs::msg::Float32 & msg);
  void timeCallback(const std_msgs::msg::Float32 & msg);
  void cycleCallback(const std_msgs::msg::Float32 & msg);
  void waypointsCallback(const std_msgs::msg::String & msg);

  void greedyJointStateCallback(const sensor_msgs::msg::JointState & msg);
  void globalJointStateCallback(const sensor_msgs::msg::JointState & msg);

  void publishSeek(float progress);
  void publishSeekFromSlider();
};

}  // namespace panda_ik_benchmark_viz_rviz_plugins

#endif  // PANDA_IK_BENCHMARK_VIZ_RVIZ_PLUGINS__IK_BENCHMARK_PLAYBACK_PANEL_HPP_
