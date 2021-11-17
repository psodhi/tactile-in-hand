// inhand variables

class gtsam::Point2;
class gtsam::Pose2;
class gtsam::Vector3;

class gtsam::Point3;
class gtsam::Pose3;
class gtsam::Vector6;

class gtsam::Values;
virtual class gtsam::noiseModel::Base;
virtual class gtsam::NonlinearFactor;
virtual class gtsam::NonlinearFactorGraph;
virtual class gtsam::NoiseModelFactor : gtsam::NonlinearFactor;

namespace inhand {

#include <inhand/dynamics/VelocitySmoothnessFactorPose3.h>
virtual class VelocitySmoothnessFactorPose3 : gtsam::NoiseModelFactor {
  VelocitySmoothnessFactor(size_t key0, size_t key1, size_t key2, const gtsam::noiseModel::Base* noiseModel);
  Vector evaluateError(const gtsam::Pose3& pose0, const gtsam::Pose3& pose1, const gtsam::Pose3& pose2) const;
};


}  // namespace inhand
