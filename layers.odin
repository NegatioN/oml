package main

// Linear layer like PyTorch - using slices for flexible sizes
Linear :: struct {
    in_features:  int,
    out_features: int,
    weight:       []f32,  // Shape: [out_features, in_features]
    bias:         []f32,  // Shape: [out_features]
//TODO should the output of the layer live in the layer to avoid re-allocation?
}

// Create a new Linear layer
linear_init :: proc(in_features, out_features: int) -> (layer: Linear) {
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
linear_forward :: proc(layer: ^Linear, input: []f32) -> []f32 {
    assert(len(input) == layer.in_features, "Input size mismatch")

    output := make([]f32, layer.out_features)

    for i in 0..<layer.out_features {
        sum := layer.bias[i]
        for j in 0..<layer.in_features {
            weight_idx := i * layer.in_features + j
            sum += input[j] * layer.weight[weight_idx]
        }
        output[i] = sum
    }

    return output
}
