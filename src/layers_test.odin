package main

import "core:testing"

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


