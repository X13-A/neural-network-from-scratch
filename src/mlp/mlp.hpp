#pragma once

#include "layers/layer.hpp"
#include "lossFunctions/lossFunctions.hpp"
#include <vector>
#include <memory>
#include <cmath>

// TODO: Store gradients and optimize outside backwards pass
class MLP {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<Eigen::VectorXf> layerOutputs;

public:
    MLP(std::vector<std::unique_ptr<Layer>> layerConfig);

    /// @brief Forward pass through the network
    /// @param inputs Input vector
    /// @param cacheEnabled Whether to cache intermediate values for backprop (disbale during inference)
    /// @return Output vector
    Eigen::VectorXf forward(const Eigen::VectorXf& inputs, bool cacheEnabled = true);

    /// @brief Backward pass through the network
    /// @param expectedOutput Expected output for loss calculation
    /// @param learningRate Learning rate for weight updates
    /// @param lossFunc Loss function to use
    void backward(const Eigen::VectorXf& expectedOutput, float learningRate, const LossFunction& lossFunc);

    size_t getLayerCount() const { return layers.size(); }

    Layer* getLayer(size_t idx) { return idx < layers.size() ? layers[idx].get() : nullptr; }
};