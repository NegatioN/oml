package main

import "core:fmt"
import "core:math"



// Linear layer like PyTorch - using slices for flexible sizes
Linear :: struct {
    in_features:  int,
    out_features: int,
    weight:       Tensor,  // Shape: [out_features, in_features]
    bias:         Tensor,  // Shape: [out_features]
}

LayerNorm :: struct {
    epsilon: f32,
    weight:       Tensor,  // Shape: [out_features, in_features]
    bias:         Tensor,  // Shape: [out_features]
}

layernorm_create :: proc(eps:f32=1e-05) -> LayerNorm {
    return LayerNorm{epsilon = eps}
}

layernorm_forward :: proc(layer: ^LayerNorm, input: Tensor, output: ^Tensor) {
    assert(input.sizes[0] == layer.weight.sizes[0], "Input size mismatch")
    assert(output.sizes[0] == layer.weight.sizes[0], "Output size mismatch") //TODO more robust size assertions
    tmp := make([]f32, 1)
    tmp[0] = mean(input)
    mean_t := make_1d_tensor(tmp)
    defer destroy_tensor(mean_t)
    tmp_t := copy_tensor(output)
    defer destroy_tensor(tmp_t)
    sub_tensors(input, mean_t, &tmp_t)


    // scary reassign
    tmp[0] = math.sqrt(stddev(input) + layer.epsilon)
    div_tensors(tmp_t, mean_t, output)
    fmt.println(output)
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
    Arange,
    LayerNorm
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

//TODO this never happens now?
layer_destroy :: proc(layer: ^Layer) {
    switch &l in layer {
    case Linear:
    case ReLU:
        // Nothing to destroy
    case Arange:
        // Nothing to destroy
    case Cat:
        // Nothing to destroy
    case LayerNorm:
    // Nothing to destroy

    }
}
