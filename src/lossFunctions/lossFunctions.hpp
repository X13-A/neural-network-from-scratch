#pragma once
#include <Eigen/Dense>

// Interface for loss functions
class LossFunction
{
public:
    virtual ~LossFunction() = default;
    virtual float loss(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const = 0;
    virtual Eigen::VectorXf derivative(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const = 0;
};

class MSE : public LossFunction
{
public:
    float loss(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const override;
    Eigen::VectorXf derivative(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const override;
};

class CrossEntropy : public LossFunction
{
public:
    float loss(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const override;
    Eigen::VectorXf derivative(const Eigen::VectorXf& output, const Eigen::VectorXf& expectedOutput) const override;
};