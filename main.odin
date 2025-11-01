#+feature dynamic-literals
package main

import "core:fmt"
import "core:time"
import "core:os"

main :: proc() {
    sw: time.Stopwatch
    time.stopwatch_start(&sw)
	fmt.println("Hello, World!")
    time.stopwatch_stop(&sw)
    fmt.println("Duration:", time.stopwatch_duration(sw))

	// Demo: Load and execute PyTorch model using full JSON parser
	test_loaded_model()

	// Demo: Manual graph executor (original demo)
	// demo_graph_executor()

	// 2D array (fixed size)
	tensor_fixed: [3][4]f32 = {
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
		{9.0, 10.0, 11.0, 12.0},
	}


	fmt.println("\n2D Fixed Array (3x4):")
	for row, i in tensor_fixed {
		fmt.println("Row", i, ":", row)
	}
    fmt.println("Element [1][2]:", tensor_fixed[1][2])

    linear := linear_init(15, 25)
    defer linear_destroy(&linear)
    /*
    model, ok := load_pytorch_model("model")
    print_weights(model)
    */
}

demo_graph_executor :: proc() {
	fmt.println("\n=== PyTorch Model Graph Executor Demo ===\n")
	
	// Manually create a simple graph (Linear -> ReLU)
	// In practice, you'd parse this from model/models/model.json
	graph := Graph{
		inputs = []Tensor_Arg{
			{as_tensor = Tensor_Name{name = "p_fc1_weight"}},
			{as_tensor = Tensor_Name{name = "p_fc1_bias"}},
			{as_tensor = Tensor_Name{name = "x"}},
		},
		outputs = []Tensor_Arg{
			{as_tensor = Tensor_Name{name = "relu"}},
		},
		nodes = []Graph_Node{
			{
				target = "torch.ops.aten.linear.default",
				inputs = []Node_Input{
					{name = "input", arg = {as_tensor = Tensor_Name{name = "x"}}, kind = 1},
					{name = "weight", arg = {as_tensor = Tensor_Name{name = "p_fc1_weight"}}, kind = 1},
					{name = "bias", arg = {as_tensor = Tensor_Name{name = "p_fc1_bias"}}, kind = 1},
				},
				outputs = []Tensor_Arg{{as_tensor = Tensor_Name{name = "linear"}}},
			},
			{
				target = "torch.ops.aten.relu.default",
				inputs = []Node_Input{
					{name = "self", arg = {as_tensor = Tensor_Name{name = "linear"}}, kind = 1},
				},
				outputs = []Tensor_Arg{{as_tensor = Tensor_Name{name = "relu"}}},
			},
		},
		tensor_values = {
			"p_fc1_weight" = Tensor_Meta{
				dtype = 7,
				sizes = []Size_Entry{{as_int = 2}, {as_int = 1}},
			},
			"p_fc1_bias" = Tensor_Meta{
				dtype = 7,
				sizes = []Size_Entry{{as_int = 2}},
			},
		},
	}
	
	// Load weights from binary files
	weights := make(map[string][]f32)
	defer {
		for _, w in weights do delete(w)
		delete(weights)
	}
	
	// Load fc1.weight (2x1 matrix)
	weight_data, w_ok := os.read_entire_file("model/data/weights/weight_0")
	if w_ok {
		defer delete(weight_data)
		num_floats := len(weight_data) / size_of(f32)
		weight_floats := transmute([]f32)weight_data
		weights["p_fc1_weight"] = make([]f32, num_floats)
		copy(weights["p_fc1_weight"], weight_floats[:num_floats])
		fmt.printf("Loaded fc1.weight: %v\n", weights["p_fc1_weight"])
	}
	
	// Load fc1.bias (length 2 vector)
	bias_data, b_ok := os.read_entire_file("model/data/weights/weight_1")
	if b_ok {
		defer delete(bias_data)
		num_floats := len(bias_data) / size_of(f32)
		bias_floats := transmute([]f32)bias_data
		weights["p_fc1_bias"] = make([]f32, num_floats)
		copy(weights["p_fc1_bias"], bias_floats[:num_floats])
		fmt.printf("Loaded fc1.bias: %v\n", weights["p_fc1_bias"])
	}
	
	// Create executor
	executor := Graph_Executor{
		graph = &graph,
		weights = weights,
		tensors = make(map[string][]f32),
	}
	defer {
		for _, t in executor.tensors do delete(t)
		delete(executor.tensors)
	}
	
	// Print graph structure
	print_graph_summary(&graph)
	
	// Test with some inputs
	fmt.println("\n=== Running Inference ===")
	test_inputs := [][]f32{
		{0.5},
		{1.0},
		{-1.0},
		{2.0},
	}
	
	for input in test_inputs {
		output, ok := execute_graph(&executor, input)
		if ok {
			fmt.printf("Input: %v -> Output: %v\n", input, output)
			delete(output)
		} else {
			fmt.printf("Failed to execute graph with input: %v\n", input)
		}
	}
}

