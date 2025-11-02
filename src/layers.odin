package main

import "core:fmt"



// Linear layer like PyTorch - using slices for flexible sizes
Linear :: struct {
    in_features:  int,
    out_features: int,
    weight:       Tensor,  // Shape: [out_features, in_features]
    bias:         Tensor,  // Shape: [out_features]
}
LayerNorm :: struct {
    in_features:  int,
    out_features: int,
    weight:       []f32,  // Shape: [out_features, in_features]
    bias:         []f32,  // Shape: [out_features]
}

// ReLU activation layer (stateless, but kept for consistency)
ReLU :: struct {
    // No parameters needed
}

Cat :: struct {
    dim: int,
}

Arange :: struct { // this is default, so only keeps end value
    end: int, //TODO datatypes
}

// Union type for all layer types
Layer :: union {
    Linear,
    ReLU,
    Cat,
    Arange
}

// Create a new Linear layer with given dimensions and weight/bias data
linear_create :: proc(in_features, out_features: int, weight, bias: Tensor) -> Linear {
    return Linear{
        in_features = in_features,
        out_features = out_features,
        weight = weight,
        bias = bias,
    }
}

// Free the layer's memory
linear_destroy :: proc(layer: ^Linear) {
    // Tensors are owned by the executor's weights map, not by the layer
    // So we don't destroy them here to avoid double-free
}

// Forward pass: output = input * weight^T + bias
// This is the core computation logic for a linear layer
linear_forward :: proc(layer: ^Linear, input: Tensor, output: ^Tensor) {
    assert(len(input.data) == layer.in_features, "Input size mismatch")
    assert(len(output.data) == layer.out_features, "Output size mismatch")

    for i in 0..<layer.out_features {
        sum := layer.bias.data[i]
        for j in 0..<layer.in_features {
            weight_idx := i * layer.in_features + j
            sum += input.data[j] * layer.weight.data[weight_idx]
        }
        output.data[i] = sum
    }
}

arange_create :: proc(val: int) -> Arange{
    return Arange{end=val}
}

cat_create :: proc(dim: int) -> Cat{
    return Cat{dim=dim}
}

arange_forward :: proc(output: []f32) {
    for i in 0..<len(output){
        output[i] = cast(f32)i
    }
}

// ReLU forward pass: output = max(0, input)
relu_forward :: proc(input: Tensor, output: ^Tensor) {
    assert(len(input.data) == len(output.data), "Input and output size mismatch")
    
    for i in 0..<len(input.data) {
        output.data[i] = max(0, input.data[i])
    }
}

// Destroy a layer (frees any allocated memory)
layer_destroy :: proc(layer: ^Layer) {
    switch &l in layer {
    case Linear:
        linear_destroy(&l)
    case ReLU:
        // Nothing to destroy
    case Arange:
        // Nothing to destroy
    case Cat:
        // Nothing to destroy

    }
}
