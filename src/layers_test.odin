package main

import "core:testing"
import "core:fmt"

// Test Linear layer forward pass
@(test)
test_linear_forward :: proc(t: ^testing.T) {
	// Setup
	weight_data := []f32{0.5, -0.5}  // 2x1 matrix
	bias_data := []f32{1.0, 2.0}
	
	weight_tensor := Tensor{
		data = weight_data,
		sizes = []int{2, 1},
		strides = []int{1, 1},
	}
	bias_tensor := Tensor{
		data = bias_data,
		sizes = []int{2},
		strides = []int{1},
	}
	
	layer := linear_create(1, 2, weight_tensor, bias_tensor)
	
	input_data := []f32{2.0}
	input_tensor := Tensor{
		data = input_data,
		sizes = []int{1},
		strides = []int{1},
	}
	
	output_data := make([]f32, 2)
	defer delete(output_data)
	output_tensor := Tensor{
		data = output_data,
		sizes = []int{2},
		strides = []int{1},
	}
	
	// Execute
	linear_forward(&layer, input_tensor, &output_tensor)
	
	// Assert: output = [2.0 * 0.5 + 1.0, 2.0 * -0.5 + 2.0] = [2.0, 1.0]
	testing.expect_value(t, output_tensor.data[0], 2.0)
	testing.expect_value(t, output_tensor.data[1], 1.0)
}

// Test ReLU activation
@(test)
test_relu_forward :: proc(t: ^testing.T) {
	input_data := []f32{-1.0, 0.0, 1.0, -5.0, 3.0}
	input_tensor := Tensor{
		data = input_data,
		sizes = []int{5},
		strides = []int{1},
	}
	
	output_data := make([]f32, len(input_data))
	defer delete(output_data)
	output_tensor := Tensor{
		data = output_data,
		sizes = []int{5},
		strides = []int{1},
	}
	
	relu_forward(input_tensor, &output_tensor)
	
	testing.expect_value(t, output_tensor.data[0], 0.0)  // max(0, -1) = 0
	testing.expect_value(t, output_tensor.data[1], 0.0)  // max(0, 0) = 0
	testing.expect_value(t, output_tensor.data[2], 1.0)  // max(0, 1) = 1
	testing.expect_value(t, output_tensor.data[3], 0.0)  // max(0, -5) = 0
	testing.expect_value(t, output_tensor.data[4], 3.0)  // max(0, 3) = 3
}

// Test Add layer with scalar (stored as single-element tensor)
@(test)
test_add_scalar :: proc(t: ^testing.T) {
	input_data := []f32{1.0, 2.0, 3.0}
	input_tensor := Tensor{data = input_data, sizes = []int{3}, strides = []int{1}}
	
	input_2_data := []f32{5.0}
	input_2_tensor := Tensor{data = input_2_data, sizes = []int{1}, strides = []int{1}}
	
	output_data := make([]f32, len(input_data))
	defer delete(output_data)
	output_tensor := Tensor{data = output_data, sizes = []int{3}, strides = []int{1}}
	
	add_forward(input_tensor, input_2_tensor, &output_tensor)
	
	// Broadcasting: single value to all elements
	testing.expect_value(t, output_tensor.data[0], 6.0)
	testing.expect_value(t, output_tensor.data[1], 7.0)
	testing.expect_value(t, output_tensor.data[2], 8.0)
}

// Test Add layer with tensor (same size)
@(test)
test_add_tensor_same_size :: proc(t: ^testing.T) {
	input_data := []f32{1.0, 2.0, 3.0}
	input_tensor := Tensor{data = input_data, sizes = []int{3}, strides = []int{1}}
	
	input_2_data := []f32{10.0, 20.0, 30.0}
	input_2_tensor := Tensor{data = input_2_data, sizes = []int{3}, strides = []int{1}}
	
	output_data := make([]f32, len(input_data))
	defer delete(output_data)
	output_tensor := Tensor{data = output_data, sizes = []int{3}, strides = []int{1}}
	
	add_forward(input_tensor, input_2_tensor, &output_tensor)
	
	testing.expect_value(t, output_tensor.data[0], 11.0)
	testing.expect_value(t, output_tensor.data[1], 22.0)
	testing.expect_value(t, output_tensor.data[2], 33.0)
}

// Test Add layer broadcasting
@(test)
test_add_broadcast :: proc(t: ^testing.T) {
	input_data := []f32{1.0, 2.0, 3.0, 4.0}  // Length 4, will broadcast [100, 200] twice
	input_tensor := Tensor{data = input_data, sizes = []int{4}, strides = []int{1}}
	
	input_2_data := []f32{100.0, 200.0}
	input_2_tensor := Tensor{data = input_2_data, sizes = []int{2}, strides = []int{1}}
	
	output_data := make([]f32, len(input_data))
	defer delete(output_data)
	output_tensor := Tensor{data = output_data, sizes = []int{4}, strides = []int{1}}
	
	add_forward(input_tensor, input_2_tensor, &output_tensor)
	
	testing.expect_value(t, output_tensor.data[0], 101.0)  // 1 + 100
	testing.expect_value(t, output_tensor.data[1], 202.0)  // 2 + 200
	testing.expect_value(t, output_tensor.data[2], 103.0)  // 3 + 100
	testing.expect_value(t, output_tensor.data[3], 204.0)  // 4 + 200
}

// Test broadcasting: larger + smaller
@(test)
test_broadcast_larger_smaller :: proc(t: ^testing.T) {
	larger := []f32{1.0, 2.0, 3.0, 4.0}
	smaller := []f32{10.0, 20.0}
	output := make([]f32, len(larger))
	defer delete(output)
	
	broadcast_op_larger_smaller(larger, smaller, output, .Add)
	
	testing.expect_value(t, output[0], 11.0)  // 1 + 10
	testing.expect_value(t, output[1], 22.0)  // 2 + 20
	testing.expect_value(t, output[2], 13.0)  // 3 + 10
	testing.expect_value(t, output[3], 24.0)  // 4 + 20
}

// Test broadcasting: smaller + larger
@(test)
test_broadcast_smaller_larger :: proc(t: ^testing.T) {
	smaller := []f32{100.0}
	larger := []f32{1.0, 2.0, 3.0}
	output := make([]f32, len(larger))
	defer delete(output)
	
	broadcast_op_smaller_larger(smaller, larger, output, .Add)
	
	testing.expect_value(t, output[0], 101.0)  // 100 + 1
	testing.expect_value(t, output[1], 102.0)  // 100 + 2
	testing.expect_value(t, output[2], 103.0)  // 100 + 3
}

// Test broadcasting with multiplication
@(test)
test_broadcast_mul :: proc(t: ^testing.T) {
	larger := []f32{2.0, 3.0, 4.0, 5.0}
	smaller := []f32{10.0}
	output := make([]f32, len(larger))
	defer delete(output)
	
	broadcast_op_larger_smaller(larger, smaller, output, .Mul)
	
	testing.expect_value(t, output[0], 20.0)   // 2 * 10
	testing.expect_value(t, output[1], 30.0)   // 3 * 10
	testing.expect_value(t, output[2], 40.0)   // 4 * 10
	testing.expect_value(t, output[3], 50.0)   // 5 * 10
}

@(test)
test_broadcast_subshape :: proc(t: ^testing.T) {
    larger := []f32{2.0, 3.0, 4.0, 5.0}
    smaller := []f32{10.0, 3.0}
    output := make([]f32, len(larger))
    defer delete(output)

    broadcast_op_larger_smaller(larger, smaller, output, .Mul)

    testing.expect_value(t, output[0], 20.0)   // 2 * 10
    testing.expect_value(t, output[1], 9.0)   // 3 * 3
    testing.expect_value(t, output[2], 40.0)   // 4 * 10
    testing.expect_value(t, output[3], 15.0)   // 5 * 3
}
