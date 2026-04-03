#include <panda_ik_benchmark_viz_rviz_plugins/ik_benchmark_playback_panel.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>

#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOptionSlider>
#include <QVBoxLayout>

#include <rclcpp/qos.hpp>

#include <rviz_common/display_context.hpp>

namespace panda_ik_benchmark_viz_rviz_plugins
{

static float clamp01(float x)
{
  if (x < 0.0F) return 0.0F;
  if (x > 1.0F) return 1.0F;
  return x;
}

static QColor globalColor()
{
  // Deep blue
  return QColor(20, 70, 220);
}

static QColor greedyColor()
{
  // Light yellow
  return QColor(235, 220, 90);
}

// -------------------------------
// IKBenchmarkTimeline
// -------------------------------
IKBenchmarkTimeline::IKBenchmarkTimeline(QWidget * parent)
: QSlider(Qt::Horizontal, parent)
{
  setRange(0, 1000);
  setSingleStep(1);
  setPageStep(10);
  setMouseTracking(true);
}

void IKBenchmarkTimeline::setTicks(const std::vector<WaypointTick> & ticks)
{
  ticks_ = ticks;
  update();
}

int IKBenchmarkTimeline::tickXpx(int tick_index) const
{
  if (tick_index < 0 || tick_index >= static_cast<int>(ticks_.size())) {
    return -1;
  }

  QStyleOptionSlider opt;
  initStyleOption(&opt);

  const QRect groove = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderGroove, this);
  const int x = groove.left() + static_cast<int>(std::round(clamp01(ticks_[tick_index].progress) * groove.width()));
  return x;
}

int IKBenchmarkTimeline::findNearestTickPx(int x_px, int tolerance_px) const
{
  int best_i = -1;
  int best_d = 999999;

  for (int i = 0; i < static_cast<int>(ticks_.size()); ++i) {
    const int x = tickXpx(i);
    if (x < 0) continue;
    const int d = std::abs(x - x_px);
    if (d < best_d) {
      best_d = d;
      best_i = i;
    }
  }

  if (best_d <= tolerance_px) {
    return best_i;
  }
  return -1;
}

void IKBenchmarkTimeline::paintEvent(QPaintEvent * ev)
{
  QSlider::paintEvent(ev);

  if (ticks_.empty()) {
    return;
  }

  QStyleOptionSlider opt;
  initStyleOption(&opt);
  const QRect groove = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderGroove, this);
  const int y_mid = groove.center().y();

  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);

  QFont f = painter.font();
  f.setPointSize(std::max(7, f.pointSize() - 1));
  painter.setFont(f);

  const bool draw_labels = static_cast<int>(ticks_.size()) <= 30;

  for (int i = 0; i < static_cast<int>(ticks_.size()); ++i) {
    const WaypointTick & t = ticks_[i];
    const int x = groove.left() + static_cast<int>(std::round(clamp01(t.progress) * groove.width()));

    const QColor c = t.is_global ? globalColor() : greedyColor();
    painter.setPen(QPen(c, 2));

    if (t.is_global) {
      painter.drawLine(QPoint(x, y_mid - 12), QPoint(x, y_mid + 2));
      painter.setBrush(c);
      painter.drawEllipse(QPoint(x, y_mid - 12), 3, 3);
      if (draw_labels) {
        painter.setPen(QPen(c, 1));
        painter.drawText(QPoint(x + 3, y_mid - 14), t.name);
      }
    } else {
      painter.drawLine(QPoint(x, y_mid - 2), QPoint(x, y_mid + 12));
      painter.setBrush(c);
      painter.drawEllipse(QPoint(x, y_mid + 12), 3, 3);
      if (draw_labels) {
        painter.setPen(QPen(c, 1));
        painter.drawText(QPoint(x + 3, y_mid + 24), t.name);
      }
    }
  }
}

void IKBenchmarkTimeline::mousePressEvent(QMouseEvent * ev)
{
  const int idx = findNearestTickPx(ev->pos().x(), 7);
  if (idx >= 0) {
    const float p = clamp01(ticks_[idx].progress);
    // Snap the slider handle to the tick.
    setValue(static_cast<int>(std::round(p * 1000.0F)));
    Q_EMIT tickClicked(p);
    ev->accept();
    return;
  }

  QSlider::mousePressEvent(ev);
}

// -------------------------------
// IKBenchmarkPlaybackPanel
// -------------------------------
IKBenchmarkPlaybackPanel::IKBenchmarkPlaybackPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * root = new QVBoxLayout(this);

  status_label_ = new QLabel("Playback: [waiting for state]");
  root->addWidget(status_label_);

  // Joint angles table (two columns: GLOBAL / GREEDY)
  joint_table_ = new QTableWidget();
  joint_table_->setColumnCount(2);
  joint_table_->setHorizontalHeaderLabels(QStringList() << "GLOBAL" << "GREEDY");
  joint_table_->horizontalHeader()->setStretchLastSection(true);
  joint_table_->verticalHeader()->setVisible(true);
  joint_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  joint_table_->setSelectionMode(QAbstractItemView::NoSelection);
  joint_table_->setFocusPolicy(Qt::NoFocus);
  joint_table_->setShowGrid(true);
  joint_table_->setMinimumHeight(220);
  root->addWidget(joint_table_);

  auto * row = new QHBoxLayout();
  pause_button_ = new QPushButton("Pause");
  row->addWidget(pause_button_);
  row->addStretch(1);
  root->addLayout(row);

  slider_ = new IKBenchmarkTimeline();
  root->addWidget(slider_);

  // Signals
  QObject::connect(pause_button_, &QPushButton::released, this, &IKBenchmarkPlaybackPanel::onTogglePause);
  QObject::connect(slider_, &QSlider::sliderPressed, this, &IKBenchmarkPlaybackPanel::onSliderPressed);
  QObject::connect(slider_, &QSlider::sliderReleased, this, &IKBenchmarkPlaybackPanel::onSliderReleased);
  QObject::connect(slider_, &QSlider::valueChanged, this, &IKBenchmarkPlaybackPanel::onSliderValueChanged);
  QObject::connect(slider_, &IKBenchmarkTimeline::tickClicked, this, &IKBenchmarkPlaybackPanel::onTickClicked);
}

IKBenchmarkPlaybackPanel::~IKBenchmarkPlaybackPanel() = default;

void IKBenchmarkPlaybackPanel::onInitialize()
{
  node_ptr_ = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!node_ptr_) {
    status_label_->setText("Playback: [ERROR: could not access RViz ROS node]");
    return;
  }

  rclcpp::Node::SharedPtr node = node_ptr_->get_raw_node();

  pause_pub_ = node->create_publisher<std_msgs::msg::Bool>(pause_topic_, 10);
  seek_pub_ = node->create_publisher<std_msgs::msg::Float32>(seek_topic_, 10);

  paused_sub_ = node->create_subscription<std_msgs::msg::Bool>(
    paused_topic_, 10,
    std::bind(&IKBenchmarkPlaybackPanel::pausedCallback, this, std::placeholders::_1));

  progress_sub_ = node->create_subscription<std_msgs::msg::Float32>(
    progress_topic_, 10,
    std::bind(&IKBenchmarkPlaybackPanel::progressCallback, this, std::placeholders::_1));

  time_sub_ = node->create_subscription<std_msgs::msg::Float32>(
    time_topic_, 10,
    std::bind(&IKBenchmarkPlaybackPanel::timeCallback, this, std::placeholders::_1));

  cycle_sub_ = node->create_subscription<std_msgs::msg::Float32>(
    cycle_topic_, 10,
    std::bind(&IKBenchmarkPlaybackPanel::cycleCallback, this, std::placeholders::_1));

  // IMPORTANT:
  // The player publishes waypoint timings as a *latched* (TRANSIENT_LOCAL) message.
  // RViz panels are often created after the player node, so if we used the default
  // VOLATILE QoS here we'd miss the one-shot publication and ticks would never appear.
  // Using TRANSIENT_LOCAL ensures we receive the last published message immediately.
  const auto qos_latched = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  waypoints_sub_ = node->create_subscription<std_msgs::msg::String>(
    waypoints_topic_, qos_latched,
    std::bind(&IKBenchmarkPlaybackPanel::waypointsCallback, this, std::placeholders::_1));

  greedy_js_sub_ = node->create_subscription<sensor_msgs::msg::JointState>(
    greedy_joint_states_topic_, 10,
    std::bind(&IKBenchmarkPlaybackPanel::greedyJointStateCallback, this, std::placeholders::_1));

  global_js_sub_ = node->create_subscription<sensor_msgs::msg::JointState>(
    global_joint_states_topic_, 10,
    std::bind(&IKBenchmarkPlaybackPanel::globalJointStateCallback, this, std::placeholders::_1));

  updateUi();
}

void IKBenchmarkPlaybackPanel::onTogglePause()
{
  paused_ = !paused_;
  std_msgs::msg::Bool msg;
  msg.data = paused_;
  if (pause_pub_) {
    pause_pub_->publish(msg);
  }
  updateUi();
}

void IKBenchmarkPlaybackPanel::onSliderPressed()
{
  dragging_ = true;
}

void IKBenchmarkPlaybackPanel::onSliderReleased()
{
  dragging_ = false;
  publishSeekFromSlider();
}

void IKBenchmarkPlaybackPanel::onSliderValueChanged(int)
{
  if (!dragging_) {
    return;
  }
  publishSeekFromSlider();
}

void IKBenchmarkPlaybackPanel::onTickClicked(float progress)
{
  // Snap + seek immediately.
  dragging_ = false;
  publishSeek(progress);
}

void IKBenchmarkPlaybackPanel::pausedCallback(const std_msgs::msg::Bool & msg)
{
  paused_ = msg.data;
  updateUi();
}

void IKBenchmarkPlaybackPanel::progressCallback(const std_msgs::msg::Float32 & msg)
{
  progress_ = clamp01(msg.data);

  if (!dragging_) {
    const int v = static_cast<int>(std::round(progress_ * 1000.0F));
    slider_->setValue(std::clamp(v, 0, 1000));
  }

  updateUi();
}

void IKBenchmarkPlaybackPanel::timeCallback(const std_msgs::msg::Float32 & msg)
{
  time_s_ = msg.data;
  updateUi();
}

void IKBenchmarkPlaybackPanel::cycleCallback(const std_msgs::msg::Float32 & msg)
{
  cycle_s_ = msg.data;
  updateUi();
}

void IKBenchmarkPlaybackPanel::waypointsCallback(const std_msgs::msg::String & msg)
{
  // Parse JSON like:
  // { "cycle_s": 5.0,
  //   "global": [{"name":"p0","t":0.0}, {"name":"p1","t":1.2}],
  //   "greedy": [{"name":"p0","t":0.0}, {"name":"p1","t":0.9}] }
  const QByteArray bytes(msg.data.c_str(), static_cast<int>(msg.data.size()));
  QJsonParseError err;
  const QJsonDocument doc = QJsonDocument::fromJson(bytes, &err);
  if (err.error != QJsonParseError::NoError || !doc.isObject()) {
    return;
  }

  const QJsonObject obj = doc.object();
  const double cycle_from_json = obj.value("cycle_s").toDouble(0.0);
  const double cycle = (cycle_s_ > 1e-6F) ? static_cast<double>(cycle_s_) : cycle_from_json;
  if (cycle <= 1e-6) {
    return;
  }

  std::vector<WaypointTick> ticks;

  auto parseArr = [&](const char * key, bool is_global) {
    const QJsonValue v = obj.value(key);
    if (!v.isArray()) {
      return;
    }
    const QJsonArray arr = v.toArray();
    for (const QJsonValue & it : arr) {
      if (!it.isObject()) continue;
      const QJsonObject o = it.toObject();
      const QString name = o.value("name").toString();
      const double t = o.value("t").toDouble(0.0);
      WaypointTick tick;
      tick.name = name;
      tick.progress = clamp01(static_cast<float>(t / cycle));
      tick.is_global = is_global;
      ticks.push_back(tick);
    }
  };

  parseArr("global", true);
  parseArr("greedy", false);

  ticks_ = ticks;
  slider_->setTicks(ticks_);
}

void IKBenchmarkPlaybackPanel::greedyJointStateCallback(const sensor_msgs::msg::JointState & msg)
{
  greedy_joint_pos_.clear();
  for (size_t i = 0; i < msg.name.size() && i < msg.position.size(); ++i) {
    greedy_joint_pos_[msg.name[i]] = msg.position[i];
  }

  if (joint_names_order_.empty()) {
    // Prefer panda_joint1..7 then fingers if available
    static const std::vector<std::string> preferred = {
      "panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7",
      "panda_finger_joint1","panda_finger_joint2"
    };
    for (const auto & j : preferred) {
      if (greedy_joint_pos_.count(j)) joint_names_order_.push_back(j);
    }
    for (const auto & kv : greedy_joint_pos_) {
      if (std::find(joint_names_order_.begin(), joint_names_order_.end(), kv.first) == joint_names_order_.end()) {
        joint_names_order_.push_back(kv.first);
      }
    }
  }

  updateJointTable();
}

void IKBenchmarkPlaybackPanel::globalJointStateCallback(const sensor_msgs::msg::JointState & msg)
{
  global_joint_pos_.clear();
  for (size_t i = 0; i < msg.name.size() && i < msg.position.size(); ++i) {
    global_joint_pos_[msg.name[i]] = msg.position[i];
  }

  if (joint_names_order_.empty()) {
    static const std::vector<std::string> preferred = {
      "panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7",
      "panda_finger_joint1","panda_finger_joint2"
    };
    for (const auto & j : preferred) {
      if (global_joint_pos_.count(j)) joint_names_order_.push_back(j);
    }
    for (const auto & kv : global_joint_pos_) {
      if (std::find(joint_names_order_.begin(), joint_names_order_.end(), kv.first) == joint_names_order_.end()) {
        joint_names_order_.push_back(kv.first);
      }
    }
  }

  updateJointTable();
}

void IKBenchmarkPlaybackPanel::publishSeek(float progress)
{
  if (!seek_pub_) {
    return;
  }
  std_msgs::msg::Float32 msg;
  msg.data = clamp01(progress);
  seek_pub_->publish(msg);
}

void IKBenchmarkPlaybackPanel::publishSeekFromSlider()
{
  const float p = clamp01(static_cast<float>(slider_->value()) / 1000.0F);
  publishSeek(p);
}

void IKBenchmarkPlaybackPanel::updateJointTable()
{
  if (joint_names_order_.empty()) {
    return;
  }

  joint_table_->setRowCount(static_cast<int>(joint_names_order_.size()));

  QStringList vheaders;
  vheaders.reserve(static_cast<int>(joint_names_order_.size()));
  for (const auto & j : joint_names_order_) {
    vheaders << QString::fromStdString(j);
  }
  joint_table_->setVerticalHeaderLabels(vheaders);

  const double rad2deg = 180.0 / 3.141592653589793;

  for (int r = 0; r < static_cast<int>(joint_names_order_.size()); ++r) {
    const std::string & jn = joint_names_order_[r];

    auto formatCell = [&](const std::map<std::string, double> & m, bool is_global) -> QTableWidgetItem * {
      auto it = m.find(jn);
      QString txt("-");
      if (it != m.end() && std::isfinite(it->second)) {
        const double deg = it->second * rad2deg;
        txt = QString::number(deg, 'f', 2);
      }
      auto * item = new QTableWidgetItem(txt);
      item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
      const QColor c = is_global ? globalColor() : greedyColor();
      item->setForeground(QBrush(c));
      return item;
    };

    joint_table_->setItem(r, 0, formatCell(global_joint_pos_, true));
    joint_table_->setItem(r, 1, formatCell(greedy_joint_pos_, false));
  }

  joint_table_->resizeColumnsToContents();
}

void IKBenchmarkPlaybackPanel::updateUi()
{
  pause_button_->setText(paused_ ? "Play" : "Pause");

  std::ostringstream ss;
  ss.setf(std::ios::fixed);
  ss.precision(2);

  if (cycle_s_ > 1e-3F) {
    ss << "Playback: "
       << (paused_ ? "PAUSED" : "PLAYING")
       << "   t=" << time_s_ << " / " << cycle_s_ << " s"
       << "   (" << static_cast<int>(std::round(progress_ * 100.0F)) << "%)"
       << "   [Click tick: jump to waypoint]";
  } else {
    ss << "Playback: " << (paused_ ? "PAUSED" : "PLAYING") << "   [waiting for cycle time]";
  }

  status_label_->setText(QString::fromStdString(ss.str()));
}

}  // namespace panda_ik_benchmark_viz_rviz_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  panda_ik_benchmark_viz_rviz_plugins::IKBenchmarkPlaybackPanel,
  rviz_common::Panel)
