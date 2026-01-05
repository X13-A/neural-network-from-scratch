#pragma once

#include <random>
#include <Eigen/Dense>

class Perceptron
{
private:
    Eigen::VectorXf weights;
    float bias;

public:
    const size_t inputSize;

    const Eigen::VectorXf& getWeights() const;

    const float getBias() const;

    void setWeights(const Eigen::VectorXf& newWeights);

    void setBias(float newBias);

    Perceptron(size_t inputSize);

    /// @brief Forward pass through the perceptron (linear only, no activation)
    /// @param inputs Input vector
    /// @param zValue Reference to store the weighted sum before activation
    /// @return Linear output (z value)
    float forward(const Eigen::VectorXf& inputs, float& zValue);
};