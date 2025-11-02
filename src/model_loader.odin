#+feature dynamic-literals
package main

import "core:fmt"
import "core:os"
import "core:encoding/json"
import "core:mem"

// Load PyTorch model from directory
load_pytorch_model_full :: proc(model_dir: string, allocator := context.allocator) -> (model: Graph_Model, weights: map[string]Tensor, ok: bool) {
	context.allocator = allocator
	
	// Load model.json
	model_json_path := fmt.tprintf("%s/models/model.json", model_dir)
	model_data, read_ok := os.read_entire_file(model_json_path)
	if !read_ok {
		fmt.printf("Failed to read model.json from: %s\n", model_json_path)
		return {}, {}, false
	}
	defer delete(model_data)
	
	// Parse JSON into Graph_Model
	parse_err := json.unmarshal(model_data, &model)
	if parse_err != nil {
		fmt.printf("Failed to parse model.json: %v\n", parse_err)
		return {}, {}, false
	}
	
	// Post-process: parse operations and expand variadic inputs/outputs
	for &node in model.graph_module.graph.nodes {
		node.operation = parse_operation(node.target)
		
		// Expand as_tensors into multiple as_tensor inputs
		expanded_inputs := expand_node_inputs(node.inputs)
		delete(node.inputs)  // Free old inputs
		node.inputs = expanded_inputs
		
		// Expand as_tensors into multiple as_tensor outputs
		expanded_outputs := expand_node_outputs(node.outputs)
		delete(node.outputs)  // Free old outputs
		node.outputs = expanded_outputs
	}
	
	// Also expand graph-level inputs and outputs
	{
		expanded := expand_node_outputs(model.graph_module.graph.inputs)
		delete(model.graph_module.graph.inputs)
		model.graph_module.graph.inputs = expanded
	}
	{
		expanded := expand_node_outputs(model.graph_module.graph.outputs)
		delete(model.graph_module.graph.outputs)
		model.graph_module.graph.outputs = expanded
	}
	
	// Load weights from binary files
	loaded_weights, weights_ok := load_weights_from_dir(model_dir)
	if !weights_ok {
		fmt.println("Failed to load weights")
		return {}, {}, false
	}
	
	return model, loaded_weights, true
}

// Load weights from the weights directory
load_weights_from_dir :: proc(model_dir: string, allocator := context.allocator) -> (weights: map[string]Tensor, ok: bool) {
	context.allocator = allocator
	
	// Use the shared helper to load weights
	config_path := fmt.tprintf("%s/data/weights/model_weights_config.json", model_dir)
	data_dir := fmt.tprintf("%s/data/weights", model_dir)
	
	return load_tensor_config(config_path, data_dir)
}

// Create executor from loaded model
create_executor_from_model :: proc(model: ^Graph_Model, weights: map[string]Tensor, model_dir: string) -> Graph_Executor {
	// Remap weights and constants using the signature
	// PyTorch exports use parameter names like "fc1.weight" and constant names like "data"
	// but the graph uses names like "p_fc1_weight" and "c_data"
	remapped_weights := make(map[string]Tensor)
	
	for input_spec in model.graph_module.signature.input_specs {
		// Handle parameters (weights/biases)
		if param, ok := input_spec.parameter.?; ok {
			param_name := param.parameter_name
			graph_name := param.arg.name
			
			if weight_tensor, found := weights[param_name]; found {
				// Deep copy to avoid double-free when maps are deleted
				remapped_weights[graph_name] = copy_tensor(weight_tensor)
			}
		}
	}
	
	// Build all constants (file-based + inline scalars) from graph
	all_constants, _ := build_constants_from_graph(&model.graph_module.graph, model_dir)
	
	// Remap file-based constants using signature
	remapped_constants := make(map[string]Tensor)
	for input_spec in model.graph_module.signature.input_specs {
		if const_info, ok := input_spec.tensor_constant.?; ok {
			const_name := const_info.tensor_constant_name
			graph_name := const_info.arg.name
			
			// Check if this constant was loaded
			if const_tensor, found := all_constants[const_name]; found {
				// Deep copy to avoid double-free when maps are deleted
				remapped_constants[graph_name] = copy_tensor(const_tensor)
				delete_key(&all_constants, const_name) // Remove from all_constants to avoid double-free
				fmt.printf("Remapped constant: %s -> %s\n", const_name, graph_name)
			}
		}
	}
	
	// Add any remaining constants (inline scalars, etc.) that don't need remapping
	for name, tensor in all_constants {
		remapped_constants[name] = tensor
	}
	delete(all_constants)
	
	// Build layers from graph nodes
	layers := build_layers_from_graph(&model.graph_module.graph, remapped_weights)
	
	executor := Graph_Executor{
		graph = &model.graph_module.graph,
		weights = remapped_weights,
		constants = remapped_constants,
		tensors = make(map[string]Tensor),
		layers = layers,
	}
	return executor
}

// Test the loaded model
test_loaded_model :: proc() {
	fmt.println("\n=== Testing Full Model Loader ===\n")
	
	// Load model and weights
	model, weights, ok := load_pytorch_model_full("model")
	if !ok {
		fmt.println("Failed to load model")
		return
	}
	// Clean up original weights map after creating executor (executor has copies)
	defer {
		for _, t in weights {
			destroy_tensor(t)
		}
		delete(weights)
	}
	
	fmt.println("\nModel loaded successfully!")
	fmt.printf("Schema version: %d.%d\n", model.schema_version.major, model.schema_version.minor)
	fmt.printf("Torch version: %s\n", model.torch_version)
	
	// Print graph info
	print_graph_summary(&model.graph_module.graph)
	
	// Create executor with constants (executor takes ownership of weight tensors)
	executor := create_executor_from_model(&model, weights, "model")
	defer {
		// Clean up runtime tensors
		for _, t in executor.tensors {
			destroy_tensor(t)
		}
		delete(executor.tensors)
		
		// Clean up constants
		for _, c in executor.constants {
			destroy_tensor(c)
		}
		delete(executor.constants)
		
		// Clean up weight tensors (executor owns these)
		for _, w in executor.weights {
			destroy_tensor(w)
		}
		delete(executor.weights)
		
		// Clean up layers
		for &layer in executor.layers do layer_destroy(&layer)
		delete(executor.layers)
	}
	fmt.printf("Schema version: %d.%d\n", model.schema_version.major, model.schema_version.minor)
	// Test inference
	fmt.println("\n=== Running Inference with Loaded Model ===")
	test_inputs := [][]f32{
		{0.5},
		{1.0},
		{-1.0},
		{2.0},
	}
	
	for input in test_inputs {
		output, exec_ok := execute_graph(&executor, input)
		if exec_ok {
			fmt.printf("Input: %v -> Output: %v\n", input, output)
			delete(output)
		} else {
			fmt.printf("Failed to execute with input: %v\n", input)
		}
	}
}
