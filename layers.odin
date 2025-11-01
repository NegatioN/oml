package main

import "core:fmt"

// Broadcasting operation type
Broadcast_Op :: enum {
    Add,
    Mul,
    Sub,
    Div,
}

// Apply binary operation with broadcasting from smaller to larger
// Precondition: len(larger) >= len(smaller)
broadcast_op_larger_smaller :: proc(larger, smaller: []f32, output: []f32, op: Broadcast_Op) {
    assert(len(output) == len(larger), "Output must match larger input size")
    assert(len(larger) >= len(smaller), "First argument must be larger or equal")
    
    larger_len := len(larger)
    smaller_len := len(smaller)
    
    if smaller_len == larger_len {
        // Same size: element-wise operation
        for i in 0..<larger_len {
            output[i] = apply_op(larger[i], smaller[i], op)
        }
    } else if smaller_len == 1 {
        // Broadcast scalar to all elements
        scalar := smaller[0]
        for i in 0..<larger_len {
            output[i] = apply_op(larger[i], scalar, op)
        }
    } else if larger_len % smaller_len == 0 {
        // Broadcast with repetition
        // E.g., smaller=[a, b] larger=[x1, x2, x3, x4] -> [op(x1,a), op(x2,b), op(x3,a), op(x4,b)]
        for i in 0..<larger_len {
            output[i] = apply_op(larger[i], smaller[i % smaller_len], op)
        }
    } else {
        panic("Incompatible broadcasting shapes")
    }
}

// Apply binary operation with broadcasting from smaller to larger (reversed operands)
// Precondition: len(smaller) <= len(larger)
broadcast_op_smaller_larger :: proc(smaller, larger: []f32, output: []f32, op: Broadcast_Op) {
    assert(len(output) == len(larger), "Output must match larger input size")
    assert(len(smaller) <= len(larger), "First argument must be smaller or equal")
    
    // Just swap the operands when calling apply_op
    larger_len := len(larger)
    smaller_len := len(smaller)
    
    if smaller_len == larger_len {
        // Same size: element-wise operation
        for i in 0..<larger_len {
            output[i] = apply_op(smaller[i], larger[i], op)
        }
    } else if smaller_len == 1 {
        // Broadcast scalar to all elements
        scalar := smaller[0]
        for i in 0..<larger_len {
            output[i] = apply_op(scalar, larger[i], op)
        }
    } else if larger_len % smaller_len == 0 {
        // Broadcast with repetition
        for i in 0..<larger_len {
            output[i] = apply_op(smaller[i % smaller_len], larger[i], op)
        }
    } else {
        panic("Incompatible broadcasting shapes")
    }
}

// Apply a binary operation to two scalars
apply_op :: proc(a, b: f32, op: Broadcast_Op) -> f32 {
    switch op {
    case .Add:
        return a + b
    case .Mul:
        return a * b
    case .Sub:
        return a - b
    case .Div:
        return a / b
    }
    return 0
}

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

// Add layer - element-wise addition
Add :: struct {
}

Arange :: struct { // this is default, so only keeps end value
    end: int, //TODO datatypes
}

// Union type for all layer types
Layer :: union {
    Linear,
    ReLU,
    Add,
    Arange
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

arange_create :: proc(val: int) -> Arange{
    return Arange{end=val}
}

arange_forward :: proc(output: []f32) {
    for i in 0..<len(output){
        output[i] = cast(f32)i
    }
}

simple_op_forward :: proc(inp: []f32, other: []f32, output: []f32, op: Broadcast_Op) {
    // Use broadcasting utility function
    input_len := len(inp)
    other_len := len(other)
    
    if input_len >= other_len {
        // inp is larger or equal, broadcast other to inp's shape
        broadcast_op_larger_smaller(inp, other, output, op)
    } else {
        // other is larger, broadcast inp to other's shape
        broadcast_op_smaller_larger(inp, other, output, op)
    }
}

add_forward :: proc(inp: []f32, other: []f32, output: []f32) {
    simple_op_forward(inp, other, output, .Add)
}
sub_forward :: proc(inp: []f32, other: []f32, output: []f32) {
    simple_op_forward(inp, other, output, .Sub)
}

// ReLU forward pass: output = max(0, input)
relu_forward :: proc(input: []f32, output: []f32) {
    assert(len(input) == len(output), "Input and output size mismatch")
    
    for i in 0..<len(input) {
        output[i] = max(0, input[i])
    }
}

// Destroy a layer (frees any allocated memory)
layer_destroy :: proc(layer: ^Layer) {
    switch &l in layer {
    case Linear:
        linear_destroy(&l)
    case ReLU:
        // Nothing to destroy for ReLU
    case Arange:
        // Nothing to destroy for ReLU
    case Add:
        // Nothing to destroy for ReLU
    }
}
