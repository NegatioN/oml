package main

// Linear layer like PyTorch - using slices for flexible sizes
Linear :: struct {
    in_features:  int,
    out_features: int,
    weight:       []f32,  // Shape: [out_features, in_features]
    bias:         []f32,  // Shape: [out_features]
}

// ReLU activation layer (stateless, but kept for consistency)
ReLU :: struct {
    // No parameters needed
}

// Union type for all layer types
Layer :: union {
    Linear,
    ReLU,
}

// Create a new Linear layer with given dimensions and weight/bias data
linear_create :: proc(in_features, out_features: int, weight, bias: []f32) -> Linear {
    return Linear{
        in_features = in_features,
        out_features = out_features,
        weight = weight,
        bias = bias,
    }
}

// Create a new Linear layer with random initialization
linear_init :: proc(in_features, out_features: int) -> Linear {
    layer: Linear
    layer.in_features = in_features
    layer.out_features = out_features

    // Allocate weight and bias
    layer.weight = make([]f32, out_features * in_features)
    layer.bias = make([]f32, out_features)

    // Initialize weights to zeros (you could use random init here)
    for i in 0..<len(layer.weight) {
        layer.weight[i] = 0.1  // Simple init
    }

    return layer
}

// Free the layer's memory
linear_destroy :: proc(layer: ^Linear) {
    delete(layer.weight)
    delete(layer.bias)
}

// Forward pass: output = input * weight^T + bias
// This is the core computation logic for a linear layer
linear_forward :: proc(layer: ^Linear, input: []f32, output: []f32) {
    assert(len(input) == layer.in_features, "Input size mismatch")
    assert(len(output) == layer.out_features, "Output size mismatch")

    for i in 0..<layer.out_features {
        sum := layer.bias[i]
        for j in 0..<layer.in_features {
            weight_idx := i * layer.in_features + j
            sum += input[j] * layer.weight[weight_idx]
        }
        output[i] = sum
    }
}

// Convenience wrapper that allocates output
linear_forward_alloc :: proc(layer: ^Linear, input: []f32) -> []f32 {
    output := make([]f32, layer.out_features)
    linear_forward(layer, input, output)
    return output
}

// Create ReLU layer (no initialization needed)
relu_create :: proc() -> ReLU {
    return ReLU{}
}

// ReLU forward pass: output = max(0, input)
relu_forward :: proc(layer: ^ReLU, input: []f32, output: []f32) {
    assert(len(input) == len(output), "Input and output size mismatch")
    
    for i in 0..<len(input) {
        output[i] = max(0, input[i])
    }
}

// Convenience wrapper that allocates output
relu_forward_alloc :: proc(layer: ^ReLU, input: []f32) -> []f32 {
    output := make([]f32, len(input))
    relu_forward(layer, input, output)
    return output
}

// Generic layer forward pass - dispatches to the correct implementation
layer_forward :: proc(layer: ^Layer, input: []f32, output: []f32) {
    switch &l in layer {
    case Linear:
        linear_forward(&l, input, output)
    case ReLU:
        relu_forward(&l, input, output)
    }
}

// Destroy a layer (frees any allocated memory)
layer_destroy :: proc(layer: ^Layer) {
    switch &l in layer {
    case Linear:
        linear_destroy(&l)
    case ReLU:
        // Nothing to destroy for ReLU
    }
}
