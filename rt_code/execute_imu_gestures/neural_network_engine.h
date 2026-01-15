// neural_network_engine.h (Header file)
#ifndef NEURAL_NETWORK_ENGINE_H
#define NEURAL_NETWORK_ENGINE_H

#include <ArduinoEigen.h>

static const int NUM_KERNELS = 10; // Number of kernel matrices in the array

class NeuralNetworkEngine {
public:

  // weights for conv1d kernels
  Eigen::MatrixXf conv1d_kernels[NUM_KERNELS]; // Array of MatrixXf objects
  // weights for batch norm - first
  Eigen::MatrixXf bn1_w;
  Eigen::MatrixXf bn1_b;
  Eigen::MatrixXf bn1_rm;
  Eigen::MatrixXf bn1_rv;
  // weights for batch norm - second
  Eigen::MatrixXf bn2_w;
  Eigen::MatrixXf bn2_b;
  Eigen::MatrixXf bn2_rm;
  Eigen::MatrixXf bn2_rv;
  // weights for linear layers
  Eigen::MatrixXf fc1_w;
  Eigen::MatrixXf fc1_b;
  Eigen::MatrixXf fc2_w;
  Eigen::MatrixXf fc2_b;

  NeuralNetworkEngine(); // Constructor declaration

  // layers inference
  Eigen::MatrixXf computeKernelFeatures(Eigen::MatrixXf input_tensor, int kernel_index);
  Eigen::MatrixXf computeConvLayer(Eigen::MatrixXf input_tensor);
  Eigen::MatrixXf computeBatchNorm(Eigen::MatrixXf input_tensor);
  Eigen::MatrixXf computeSecondBatchNorm(Eigen::MatrixXf input_tensor);
  Eigen::MatrixXf computeFirstLinearLayer(Eigen::MatrixXf input_tensor);
  Eigen::MatrixXf computeSecondLinearLayer(Eigen::MatrixXf input_tensor);

  // main inference functions - return class index
  uint8_t predict(Eigen::MatrixXf input_tensor);
  uint8_t maxClassIndex(Eigen::MatrixXf logits);

  // activation functions
  Eigen::MatrixXf applyRELU(Eigen::MatrixXf input_tensor);
};

#endif // NEURAL_NETWORK_ENGINE_H