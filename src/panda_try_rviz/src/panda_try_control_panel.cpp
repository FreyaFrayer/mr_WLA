#include <panda_try_rviz/panda_try_control_panel.hpp>

#include <algorithm>
#include <cmath>
#include <functional>

#include <QGridLayout>
#include <QHBoxLayout>
#include <QSignalBlocker>
#include <QVBoxLayout>

#include <rviz_common/display_context.hpp>

namespace panda_try_rviz
{

namespace
{

// Approximate Panda joint limits (deg).
constexpr std::array<double, 7> kJointMinDeg = {
  -166.0, -101.0, -166.0, -176.0, -166.0, -1.0, -166.0
};
constexpr std::array<double, 7> kJointMaxDeg = {
  166.0, 101.0, 166.0, -4.0, 166.0, 215.0, 166.0
};

// A typical ready-ish pose used as startup state.
constexpr std::array<double, 7> kInitDeg = {
  0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0
};

}  // namespace

PandaTryControlPanel::PandaTryControlPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * root = new QVBoxLayout(this);

  title_label_ = new QLabel("Panda Try Control");
  title_label_->setStyleSheet("font-weight: 600;");
  root->addWidget(title_label_);

  collision_label_ = new QLabel();
  refreshCollisionLabel(false);
  root->addWidget(collision_label_);

  auto * grid = new QGridLayout();
  grid->setColumnStretch(1, 1);

  for (int i = 0; i < 7; ++i) {
    auto * name_label = new QLabel(QString("J%1").arg(i + 1));
    grid->addWidget(name_label, i, 0);

    auto * slider = new QSlider(Qt::Horizontal);
    slider->setRange(
      static_cast<int>(std::round(jointMinDeg(i) * 10.0)),
      static_cast<int>(std::round(jointMaxDeg(i) * 10.0)));
    slider->setSingleStep(1);     // 0.1 deg
    slider->setPageStep(20);      // 2.0 deg

    const int init_v = static_cast<int>(std::round(clampDeg(kInitDeg[static_cast<size_t>(i)], i) * 10.0));
    slider->setValue(init_v);

    auto * value_label = new QLabel();
    value_label->setMinimumWidth(70);
    value_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    auto * angle_input = new QDoubleSpinBox();
    angle_input->setDecimals(1);
    angle_input->setRange(jointMinDeg(i), jointMaxDeg(i));
    angle_input->setSingleStep(0.1);
    angle_input->setSuffix(" deg");
    angle_input->setKeyboardTracking(false);
    angle_input->setAlignment(Qt::AlignRight);
    angle_input->setMinimumWidth(95);
    angle_input->setValue(static_cast<double>(slider->value()) / 10.0);

    sliders_[static_cast<size_t>(i)] = slider;
    value_labels_[static_cast<size_t>(i)] = value_label;
    angle_inputs_[static_cast<size_t>(i)] = angle_input;
    joint_positions_rad_[static_cast<size_t>(i)] = (static_cast<double>(slider->value()) / 10.0) * kDegToRad;

    refreshJointValueLabel(i);

    grid->addWidget(slider, i, 1);
    grid->addWidget(value_label, i, 2);
    grid->addWidget(angle_input, i, 3);

    QObject::connect(slider, &QSlider::valueChanged, this, [this, i](int value) {
      onJointSliderChanged(i, value);
    });

    QObject::connect(
      angle_input, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
      [this, i](double value_deg) {
        onJointInputChanged(i, value_deg);
      });
  }

  root->addLayout(grid);

  auto * btn_row = new QHBoxLayout();
  save_button_ = new QPushButton("Save Current EE Point");
  btn_row->addWidget(save_button_);
  btn_row->addStretch(1);
  root->addLayout(btn_row);

  QObject::connect(save_button_, &QPushButton::released, this, &PandaTryControlPanel::onSavePointClicked);
}

PandaTryControlPanel::~PandaTryControlPanel() = default;

void PandaTryControlPanel::onInitialize()
{
  node_ptr_ = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!node_ptr_) {
    title_label_->setText("Panda Try Control [ERROR: no RViz ROS node]");
    return;
  }

  rclcpp::Node::SharedPtr node = node_ptr_->get_raw_node();

  joint_cmd_pub_ = node->create_publisher<sensor_msgs::msg::JointState>(
    "/panda_try/command_joint_states", 10);
  save_point_pub_ = node->create_publisher<std_msgs::msg::Empty>(
    "/panda_try/save_ee_point", 10);

  collision_sub_ = node->create_subscription<std_msgs::msg::Bool>(
    "/panda_try/self_collision", 10,
    std::bind(&PandaTryControlPanel::onCollisionStatus, this, std::placeholders::_1));

  publishCurrentJointState();
}

double PandaTryControlPanel::jointMinDeg(int idx)
{
  if (idx < 0 || idx >= static_cast<int>(kJointMinDeg.size())) {
    return -180.0;
  }
  return kJointMinDeg[static_cast<size_t>(idx)];
}

double PandaTryControlPanel::jointMaxDeg(int idx)
{
  if (idx < 0 || idx >= static_cast<int>(kJointMaxDeg.size())) {
    return 180.0;
  }
  return kJointMaxDeg[static_cast<size_t>(idx)];
}

double PandaTryControlPanel::clampDeg(double v_deg, int idx)
{
  return std::clamp(v_deg, jointMinDeg(idx), jointMaxDeg(idx));
}

void PandaTryControlPanel::onJointSliderChanged(int idx, int slider_value)
{
  if (idx < 0 || idx >= 7) {
    return;
  }

  const double deg = clampDeg(static_cast<double>(slider_value) / 10.0, idx);
  auto * angle_input = angle_inputs_[static_cast<size_t>(idx)];
  if (angle_input) {
    QSignalBlocker blocker(angle_input);
    angle_input->setValue(deg);
  }

  joint_positions_rad_[static_cast<size_t>(idx)] = deg * kDegToRad;
  refreshJointValueLabel(idx);
  publishCurrentJointState();
}

void PandaTryControlPanel::onJointInputChanged(int idx, double input_deg)
{
  if (idx < 0 || idx >= 7) {
    return;
  }

  const double clamped_deg = clampDeg(input_deg, idx);
  const int slider_value = static_cast<int>(std::round(clamped_deg * 10.0));
  auto * slider = sliders_[static_cast<size_t>(idx)];
  if (!slider) {
    return;
  }

  {
    QSignalBlocker blocker(slider);
    slider->setValue(slider_value);
  }

  const double snapped_deg = static_cast<double>(slider->value()) / 10.0;
  auto * angle_input = angle_inputs_[static_cast<size_t>(idx)];
  if (angle_input) {
    QSignalBlocker blocker(angle_input);
    angle_input->setValue(snapped_deg);
  }

  joint_positions_rad_[static_cast<size_t>(idx)] = snapped_deg * kDegToRad;
  refreshJointValueLabel(idx);
  publishCurrentJointState();
}

void PandaTryControlPanel::onSavePointClicked()
{
  if (!save_point_pub_) {
    return;
  }

  std_msgs::msg::Empty msg;
  save_point_pub_->publish(msg);
}

void PandaTryControlPanel::onCollisionStatus(const std_msgs::msg::Bool & msg)
{
  refreshCollisionLabel(msg.data);
}

void PandaTryControlPanel::publishCurrentJointState()
{
  if (!joint_cmd_pub_) {
    return;
  }

  sensor_msgs::msg::JointState msg;
  msg.header.stamp = node_ptr_->get_raw_node()->now();

  msg.name.reserve(7);
  msg.position.reserve(7);

  for (int i = 0; i < 7; ++i) {
    msg.name.emplace_back(kJointNames[i]);
    msg.position.push_back(joint_positions_rad_[static_cast<size_t>(i)]);
  }

  joint_cmd_pub_->publish(msg);
}

void PandaTryControlPanel::refreshCollisionLabel(bool in_collision)
{
  if (in_collision) {
    collision_label_->setText("Self-Collision: YES");
    collision_label_->setStyleSheet("color: #ff4b4b; font-weight: 600;");
  } else {
    collision_label_->setText("Self-Collision: NO");
    collision_label_->setStyleSheet("color: #57d957; font-weight: 600;");
  }
}

void PandaTryControlPanel::refreshJointValueLabel(int idx)
{
  if (idx < 0 || idx >= 7) {
    return;
  }

  auto * value_label = value_labels_[static_cast<size_t>(idx)];
  if (!value_label) {
    return;
  }

  const double deg = joint_positions_rad_[static_cast<size_t>(idx)] * kRadToDeg;
  value_label->setText(QString::number(deg, 'f', 1) + " deg");
}

}  // namespace panda_try_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(panda_try_rviz::PandaTryControlPanel, rviz_common::Panel)
