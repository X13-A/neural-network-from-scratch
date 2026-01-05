#include <iostream>
#include <vector>
#include <array>
#include <functional>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <random>

#include "perceptron/perceptron.hpp"
#include "mlp/mlp.hpp"
#include "layers/denseLayer.hpp"
#include "layers/activationLayers.hpp"
#include "lossFunctions/lossFunctions.hpp"

#pragma region MNIST_PARSING
// NOTE: MNIST parsing code is AI-generated
uint32_t read_uint32_be(std::ifstream& file)
{
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}
std::vector<Eigen::VectorXf> loadMNISTImages(const std::string& filename, int maxSamples = -1)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MNIST image file: " + filename);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    uint32_t numImages = read_uint32_be(file);
    uint32_t numRows = read_uint32_be(file);
    uint32_t numCols = read_uint32_be(file);

    int imagesToRead = (maxSamples == -1) ? numImages : std::min((uint32_t)maxSamples, numImages);
    std::vector<Eigen::VectorXf> images;

    for (int i = 0; i < imagesToRead; ++i) {
        Eigen::VectorXf image(numRows * numCols);
        unsigned char pixel;
        for (uint32_t j = 0; j < numRows * numCols; ++j) {
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image(j) = pixel / 255.0f; // Normalize to [0, 1]
        }
        images.push_back(image);
    }

    file.close();
    return images;
}
std::vector<int> loadMNISTLabels(const std::string& filename, int maxSamples = -1)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MNIST label file: " + filename);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    uint32_t numLabels = read_uint32_be(file);
    int labelsToRead = (maxSamples == -1) ? numLabels : std::min((uint32_t)maxSamples, numLabels);
    std::vector<int> labels;

    for (int i = 0; i < labelsToRead; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(label);
    }

    file.close();
    return labels;
}
Eigen::VectorXf labelToOneHot(int label, int numClasses = 10)
{
    Eigen::VectorXf oneHot = Eigen::VectorXf::Zero(numClasses);
    oneHot(label) = 1.0f;
    return oneHot;
}
int getPredictedDigit(const Eigen::VectorXf& output)
{
    int predicted = 0;
    float maxVal = output(0);
    for (int i = 1; i < output.size(); ++i) {
        if (output(i) > maxVal) {
            maxVal = output(i);
            predicted = i;
        }
    }
    return predicted;
}
#pragma endregion

void trainMNISTDigitClassifier()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "MNIST Digit Classification" << std::endl;
    std::cout << "========================================" << std::endl;

    try 
    {
        // Load MNIST dataset
        std::cout << "\nLoading MNIST training data..." << std::endl;
        std::string trainImagesPath = "data/train-images-idx3-ubyte";
        std::string trainLabelsPath = "data/train-labels-idx1-ubyte";
        std::string testImagesPath = "data/t10k-images-idx3-ubyte";
        std::string testLabelsPath = "data/t10k-labels-idx1-ubyte";

        // Use N images per epoch
        int imagesPerEpoch = 500;
        auto trainImages = loadMNISTImages(trainImagesPath, imagesPerEpoch);
        auto trainLabels = loadMNISTLabels(trainLabelsPath, imagesPerEpoch);
        auto testImages = loadMNISTImages(testImagesPath, 100);
        auto testLabels = loadMNISTLabels(testLabelsPath, 100);

        std::cout << "Loaded " << trainImages.size() << " training samples" << std::endl;
        std::cout << "Loaded " << testImages.size() << " test samples" << std::endl;

        std::vector<std::unique_ptr<Layer>> layers;
        layers.push_back(std::make_unique<DenseLayer>(784, 128));
        layers.push_back(std::make_unique<DenseLayer>(128, 64));
        layers.push_back(std::make_unique<ReLULayer>());
        layers.push_back(std::make_unique<DenseLayer>(64, 10));
        layers.push_back(std::make_unique<SoftmaxLayer>());

        MLP mlp(std::move(layers));

        float learningRate = 0.05f;
        int epochs = 25;

        std::cout << "\nTraining MLP for digit classification..." << std::endl;
        std::cout << "Learning Rate: " << learningRate << ", Epochs: " << epochs << std::endl;
        std::cout << "Epoch\t\tAvg CrossEntropy\tAccuracy" << std::endl;
        std::cout << "-----\t\t--------------\t--------" << std::endl;

        // Create indices for shuffling
        std::vector<int> indices(trainImages.size());
        for (int i = 0; i < trainImages.size(); ++i) 
        {
            indices[i] = i;
        }

        // Training loop
        for (int epoch = 0; epoch < epochs; ++epoch) 
        {
            float totalLoss = 0.0f;
            int correctPredictions = 0;
            int totalProcessed = 0;

            // Shuffle indices for each epoch
            std::shuffle(indices.begin(), indices.end(), std::default_random_engine(epoch));

            for (int i = 0; i < imagesPerEpoch; ++i) 
            {
                int sampleIdx = indices[i];
                Eigen::VectorXf output = mlp.forward(trainImages[sampleIdx], true);
                Eigen::VectorXf target = labelToOneHot(trainLabels[sampleIdx], 10);
                
                CrossEntropy lossFunc;
                float loss = lossFunc.loss(output, target);
                mlp.backward(target, learningRate, lossFunc);
                totalLoss += loss;

                // Check if prediction is correct
                int predicted = getPredictedDigit(output);
                if (predicted == trainLabels[sampleIdx]) 
                {
                    correctPredictions++;
                }
                totalProcessed++;
            }

            float avgLoss = totalLoss / totalProcessed;
            float accuracy = (100.0f * correctPredictions) / totalProcessed;
            std::cout << epoch << "\t\t" << std::fixed << avgLoss << "\t" << accuracy << "%" << std::endl;
        }

        // Test on test set
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing on Test Set" << std::endl;
        std::cout << "========================================" << std::endl;

        int testCorrect = 0;
        float testLoss = 0.0f;
        for (int i = 0; i < testImages.size(); ++i) 
        {
            Eigen::VectorXf output = mlp.forward(testImages[i], false);
            int predicted = getPredictedDigit(output);
            int actual = testLabels[i];
            Eigen::VectorXf target = labelToOneHot(actual, 10);
            CrossEntropy lossFunc;
            testLoss += lossFunc.loss(output, target);

            if (predicted == actual) 
            {
                testCorrect++;
            }

            // Print first predictions
            if (i < 20) 
            {
                std::cout << "Sample " << i << ": Predicted = " << predicted 
                          << ", Actual = " << actual;
                if (predicted == actual) 
                {
                    std::cout << " [CORRECT]" << std::endl;
                }
                else
                {
                    std::cout << " [WRONG]" << std::endl;
                }
            }
        }

        float testAccuracy = (100.0f * testCorrect) / testImages.size();
        float avgTestLoss = testLoss / testImages.size();
        std::cout << "\nTest Accuracy: " << testAccuracy << "% (" << testCorrect 
                  << "/" << testImages.size() << ")" << std::endl;
        std::cout << "Test Avg CrossEntropy: " << avgTestLoss << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << "\nNote: MNIST dataset files are required at:" << std::endl;
        std::cout << "  - data/train-images-idx3-ubyte" << std::endl;
        std::cout << "  - data/train-labels-idx1-ubyte" << std::endl;
        std::cout << "  - data/t10k-images-idx3-ubyte" << std::endl;
        std::cout << "  - data/t10k-labels-idx1-ubyte" << std::endl;
    }
}

int main() {
    std::cout << "Neural Network from Scratch" << std::endl;
    std::cout << "===================================" << std::endl << std::endl;
    trainMNISTDigitClassifier();
    std::cout << std::endl;
    return 0;
}
