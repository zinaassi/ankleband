// neural_network_engine.cpp (Source file)
#include "neural_network_engine.h"
#include "model_weights_c1d_jul10.h"

const int nn_input_length = 60;
const int kernel_output_length = nn_input_length / conv1d_1_weight_kernel_size;
const float bn_eps = 1e-5;

NeuralNetworkEngine::NeuralNetworkEngine() {

  // allocate sizes for weight matrices
  for (int i = 0; i < NUM_KERNELS; i++) {
    conv1d_kernels[i].resize(conv1d_1_weight_num_channels, conv1d_1_weight_kernel_size); // initialize each kernel matrix
  }
  bn1_w.resize(bn1_size, 1);
  bn1_b.resize(bn1_size, 1);
  bn1_rm.resize(bn1_size, 1);
  bn1_rv.resize(bn1_size, 1);
  bn2_w.resize(bn2_size, 1);
  bn2_b.resize(bn2_size, 1);
  bn2_rm.resize(bn2_size, 1);
  bn2_rv.resize(bn2_size, 1);
  fc1_w.resize(fc_layers_0_weight_num_rows, fc_layers_0_weight_num_cols);
  fc1_b.resize(fc_layers_0_bias_size, 1);
  fc2_w.resize(fc_layers_3_weight_num_rows, fc_layers_3_weight_num_cols);
  fc2_b.resize(fc_layers_3_bias_size, 1);

  // set values
  // conv1d kernels
  for (int i = 0; i < conv1d_1_weight_num_kernels; i++) {
    for (int j = 0; j < conv1d_1_weight_num_channels; j++) {
      for (int k = 0; k < conv1d_1_weight_kernel_size; k++) {
        conv1d_kernels[i](j, k) = conv1d_1_weight[i][j][k];
      }
    }
  }
  // batch morm - first
  for (int i = 0; i < bn1_size; i++) {
    bn1_w(i, 0) = bn1_weight[i];
    bn1_b(i, 0) = bn1_bias[i];
    bn1_rm(i, 0) = bn1_running_mean[i];
    bn1_rv(i, 0) = bn1_running_var[i];
  }
  // batch morm - second
  for (int i = 0; i < bn2_size; i++) {
    bn2_w(i, 0) = bn2_weight[i];
    bn2_b(i, 0) = bn2_bias[i];
    bn2_rm(i, 0) = bn2_running_mean[i];
    bn2_rv(i, 0) = bn2_running_var[i];
  }
  for (int i = 0; i < fc_layers_0_weight_num_rows; i++) {
    for (int j = 0; j < fc_layers_0_weight_num_cols; j++) {
      fc1_w(i, j) = fc_layers_0_weight[i][j];
    }
  }
  for (int i = 0; i < fc_layers_0_bias_size; i++) {
    fc1_b(i, 0) = fc_layers_0_bias[i];
  }
  for (int i = 0; i < fc_layers_3_weight_num_rows; i++) {
    for (int j = 0; j < fc_layers_3_weight_num_cols; j++) {
      fc2_w(i, j) = fc_layers_3_weight[i][j];
    }
  }
  for (int i = 0; i < fc_layers_3_bias_size; i++) {
    fc2_b(i, 0) = fc_layers_3_bias[i];
  }
}

// compute the result for a specific kernel on the input tensor
Eigen::MatrixXf NeuralNetworkEngine::computeKernelFeatures(Eigen::MatrixXf input_tensor, int kernel_index) {

  // create output vector for a single kernel
  Eigen::VectorXf kernel_result(kernel_output_length);

  // perform 1D-convolution
  for (int i = 0; i < kernel_output_length; i++) {
    int start_col = i * conv1d_1_weight_kernel_size; // starting column for the current block
    Eigen::MatrixXf block = input_tensor.block(0, start_col, 6, 3); // extract the block

    // perform the dot product (or convolution operation)
    kernel_result(i) = (block.array() * conv1d_kernels[kernel_index].array()).sum(); // element-wise multiplication and sum
  }

  return kernel_result;
}

Eigen::MatrixXf NeuralNetworkEngine::computeConvLayer(Eigen::MatrixXf input_tensor) {

  // create an array of output vectors (for each kernel)
  Eigen::VectorXf kernels_results[conv1d_1_weight_num_kernels];

  // perform the convolution for each kernel
  for (int k = 0; k < conv1d_1_weight_num_kernels; k++) {
    kernels_results[k].resize(kernel_output_length); // Important: Resize each vector!
    kernels_results[k] = computeKernelFeatures(input_tensor, k);
  }

  // concatenate the output vectors into a single vector
  Eigen::VectorXf kernels_result(conv1d_1_weight_num_kernels * kernel_output_length);
  for (int i = 0; i < conv1d_1_weight_num_kernels; i++) {
    kernels_result.segment(i * kernel_output_length, kernel_output_length) = kernels_results[i];
  }

  return kernels_result;
}

Eigen::MatrixXf NeuralNetworkEngine::computeBatchNorm(Eigen::MatrixXf input_tensor) {

  // normalize vector
  Eigen::VectorXf layer_result = input_tensor - bn1_rm;
  layer_result = layer_result.array() / (bn1_rv.array() + bn_eps).array().sqrt();

  // scale and shift
  layer_result = (bn1_w.array() * layer_result.array()) + bn1_b.array();

  return layer_result;
}
Eigen::MatrixXf NeuralNetworkEngine::computeSecondBatchNorm(Eigen::MatrixXf input_tensor) {

  // normalize vector
  Eigen::VectorXf layer_result = input_tensor - bn2_rm;
  layer_result = layer_result.array() / (bn2_rv.array() + bn_eps).array().sqrt();

  // scale and shift
  layer_result = (bn2_w.array() * layer_result.array()) + bn2_b.array();

  return layer_result;
}

Eigen::MatrixXf NeuralNetworkEngine::computeFirstLinearLayer(Eigen::MatrixXf input_tensor) {

  // compute Wx + b
  Eigen::VectorXf layer_result = fc1_w * input_tensor + fc1_b; // Result should be 64x200 * 200x1 + 64x1 = 64x1

  return layer_result;
}

Eigen::MatrixXf NeuralNetworkEngine::computeSecondLinearLayer(Eigen::MatrixXf input_tensor) {

  // compute Wx + b
  Eigen::VectorXf layer_result = fc2_w * input_tensor + fc2_b; // Result should be 64x64 * 64x1 + 64x1 = 64x1

  return layer_result;
}

Eigen::MatrixXf NeuralNetworkEngine::applyRELU(Eigen::MatrixXf input_tensor) {
  
  return input_tensor.array().max(0.0f);
}


uint8_t NeuralNetworkEngine::predict(Eigen::MatrixXf input_tensor) {

  // feed forward
  Eigen::MatrixXf kernel_res = computeConvLayer(input_tensor);
  Eigen::MatrixXf bnlayer_res = computeBatchNorm(kernel_res);
  Eigen::MatrixXf relu_res = applyRELU(bnlayer_res);
  Eigen::MatrixXf fc1_res = computeFirstLinearLayer(relu_res);
  Eigen::MatrixXf bn2layer_res = computeSecondBatchNorm(fc1_res);
  Eigen::MatrixXf fc1_relu_res = applyRELU(bn2layer_res);
  Eigen::MatrixXf logits = computeSecondLinearLayer(fc1_relu_res);


  // extract max index for class probability
  int argmax_index = maxClassIndex(logits);

  // for debug - print prediction and chosen class
  // Print the result.
  Serial.print("Predictions: ");
  for (int i = 0; i < 5; i++) {
    Serial.print(logits(i));
    Serial.print(" ");
  }
  Serial.print(". Class: ");
  Serial.print(argmax_index);
  Serial.println(" ");

  return argmax_index;
}

uint8_t NeuralNetworkEngine::maxClassIndex(Eigen::MatrixXf logits) {
  
  int argmax_index = 0;
  float max_value = -std::numeric_limits<float>::infinity(); // initialize to negative infinity

  // iterate over the logits vector
  for (int i = 0; i < 5; ++i) {
    if (logits(i) > max_value) {
      max_value = logits(i);
      argmax_index = i;
    }
  }
  return argmax_index;
}