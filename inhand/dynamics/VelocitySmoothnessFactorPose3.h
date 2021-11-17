#ifndef VELOCITY_SMOOTHNESS_FACTOR_H
#define VELOCITY_SMOOTHNESS_FACTOR_H

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace inhand {

class VelocitySmoothnessFactorPose3 : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {
private:
  bool useAnalyticJacobians_;

public:
  VelocitySmoothnessFactorPose3(gtsam::Key key0, gtsam::Key key1, gtsam::Key key2, const gtsam::SharedNoiseModel &model = nullptr)
      : NoiseModelFactor3(model, key0, key1, key2), useAnalyticJacobians_(false) {}

  gtsam::Vector6 smoothnessError(const gtsam::Pose3 &pose0, const gtsam::Pose3 &pose1, const gtsam::Pose3 &pose2) const
  {
    gtsam::Pose3 pose01 = pose0.between(pose1);
    gtsam::Pose3 pose12 = pose1.between(pose2);

    // gtsam::Vector3 errRot, errTrans;
    // errRot << pose01.rotation().rpy() - pose12.rotation().rpy();
    // errTrans << pose01.translation() - pose12.translation();
    // gtsam::Vector6 errVec;
    // errVec << errRot, errTrans;

    gtsam::Pose3 poseDiff = pose01.between(pose12);
    gtsam::Vector6 errVec = gtsam::Pose3::Logmap(poseDiff);

    // std::cout << "[VelocitySmoothnessFactorPose3] errVec: " << errVec << std::endl;

    return errVec;
  }

  gtsam::Vector evaluateError(const gtsam::Pose3& pose0, const gtsam::Pose3& pose1, const gtsam::Pose3& pose2,
                              const boost::optional<gtsam::Matrix&> H1 = boost::none,
                              const boost::optional<gtsam::Matrix&> H2 = boost::none,
                              const boost::optional<gtsam::Matrix&> H3 = boost::none) const {

    gtsam::Vector6 errVec = VelocitySmoothnessFactorPose3::smoothnessError(pose0, pose1, pose2);
    gtsam::Matrix J1, J2, J3;

    if (useAnalyticJacobians_) {
    } else {
      J1 = gtsam::numericalDerivative11<gtsam::Vector6, gtsam::Pose3>(boost::bind(&VelocitySmoothnessFactorPose3::smoothnessError, this, _1, pose1, pose2), pose0);
      J2 = gtsam::numericalDerivative11<gtsam::Vector6, gtsam::Pose3>(boost::bind(&VelocitySmoothnessFactorPose3::smoothnessError, this, pose0, _1, pose2), pose1);
      J3 = gtsam::numericalDerivative11<gtsam::Vector6, gtsam::Pose3>(boost::bind(&VelocitySmoothnessFactorPose3::smoothnessError, this, pose0, pose1, _1), pose2);
    }

    if (H1) *H1 = J1;
    if (H2) *H2 = J2;
    if (H3) *H3 = J3;

    return errVec;
  }
};

}  // namespace inhand

#endif