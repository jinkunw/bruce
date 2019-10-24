#include "bruce_exploration/planner/nf_exploration_server.h"
#include "bruce_exploration/planner/nbv_exploration_server.h"
#include "bruce_exploration/planner/heuristic_exploration_server.h"
#include "bruce_exploration/planner/em_exploration_server.h"

using namespace bruce_exploration::planner;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "exploration_server");
  ros::NodeHandle nh("~");

  std::string algorithm;
  nh.getParam("algorithm", algorithm);

  if (algorithm == "nf")
  {
    NFExplorationServer server(nh);
    ROS_INFO("Nearest-frontier exploration server started...");
    ros::spin();
  }
  else if (algorithm == "nbv")
  {
    NBVExplorationServer server(nh);
    ROS_INFO("Next-best-view exploration server started...");
    ros::spin();
  }
  else if (algorithm == "heuristic")
  {
    HeuristicExplorationServer server(nh);
    ROS_INFO("Heuristic exploration server started...");
    ros::spin();
  }
  else if (algorithm == "em")
  {
    EMExplorationServer server(nh);
    ROS_INFO("EM exploration server started...");
    ros::spin();
  }

  return 0;
}