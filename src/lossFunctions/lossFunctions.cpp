#include "lossFunctions/lossFunctions.hpp"

float MSE::loss(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const
{
    Eigen::VectorXf diff = output - expectedOutput;
    return diff.squaredNorm() / output.size();
}

Eigen::VectorXf MSE::derivative(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const
{
    return 2.0f * (output - expectedOutput) / output.size();
}

float CrossEntropy::loss(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const
{
    const float epsilon = 1e-7f;
    Eigen::VectorXf clipped = output.cwiseMax(epsilon).cwiseMin(1.0f - epsilon);
    return -(expectedOutput.array() * clipped.array().log()).sum() / output.size();
}

Eigen::VectorXf CrossEntropy::derivative(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const
{
    return (output - expectedOutput) / output.size();
}