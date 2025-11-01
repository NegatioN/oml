#+feature dynamic-literals
package main

import "core:fmt"
import "core:os"
import "core:encoding/json"
import "core:mem"

// Load PyTorch model from directory
load_pytorch_model_full :: proc(model_dir: string, allocator := context.allocator) -> (model: Graph_Model, weights: map[string][]f32, ok: bool) {
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
    for &n in model.graph_module.graph.nodes {
        n.operation = parse_operation(n.target)
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
load_weights_from_dir :: proc(model_dir: string, allocator := context.allocator) -> (weights: map[string][]f32, ok: bool) {
	context.allocator = allocator
	
	weights = make(map[string][]f32)
	
	// Read weights config
	config_path := fmt.tprintf("%s/data/weights/model_weights_config.json", model_dir)
	config_data, read_ok := os.read_entire_file(config_path)
	if !read_ok {
		fmt.printf("Failed to read weights config: %s\n", config_path)
		return weights, false
	}
	defer delete(config_data)
	
	// Parse weights config
	Weights_Config_Root :: struct {
		config: map[string]Weight_Config,
	}
	
	Weight_Config :: struct {
		path_name:   string,
		is_param:    bool,
		use_pickle:  bool,
		tensor_meta: Tensor_Meta,
	}
	
	config: Weights_Config_Root
	parse_err := json.unmarshal(config_data, &config)
	if parse_err != nil {
		fmt.printf("Failed to parse weights config: %v\n", parse_err)
		return weights, false
	}
	defer delete(config.config)
	
	// Load each weight file
	for param_name, weight_info in config.config {
		weight_path := fmt.tprintf("%s/data/weights/%s", model_dir, weight_info.path_name)
		weight_data, w_ok := os.read_entire_file(weight_path)
		if !w_ok {
			fmt.printf("Failed to read weight file: %s\n", weight_path)
			continue
		}
		defer delete(weight_data)
		
		// Convert bytes to floats
		num_floats := len(weight_data) / size_of(f32)
		weight_floats := transmute([]f32)weight_data
		
		// Store with parameter name
		weights[param_name] = make([]f32, num_floats)
		copy(weights[param_name], weight_floats[:num_floats])
		
		fmt.printf("Loaded %s: %v floats from %s\n", param_name, num_floats, weight_info.path_name)
	}
	
	return weights, true
}

// Create executor from loaded model
create_executor_from_model :: proc(model: ^Graph_Model, weights: map[string][]f32) -> Graph_Executor {
	// Remap weights using the signature
	// PyTorch exports use parameter names like "fc1.weight"
	// but the graph uses names like "p_fc1_weight"
	remapped_weights := make(map[string][]f32)
	
	for input_spec in model.graph_module.signature.input_specs {
		if param, ok := input_spec.parameter.?; ok {
			param_name := param.parameter_name
			graph_name := param.arg.name
			
			if weight_data, found := weights[param_name]; found {
				// COPY the weight data to the graph name (don't alias!)
				copied_weight := make([]f32, len(weight_data))
				copy(copied_weight, weight_data)
				remapped_weights[graph_name] = copied_weight
			}
		}
	}
	
	// Build layers from graph nodes
	layers := build_layers_from_graph(&model.graph_module.graph, remapped_weights)
	
	executor := Graph_Executor{
		graph = &model.graph_module.graph,
		weights = remapped_weights,
		tensors = make(map[string][]f32),
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
	defer {
		for _, w in weights do delete(w)
		delete(weights)
	}
	
	fmt.println("\nModel loaded successfully!")
	fmt.printf("Schema version: %d.%d\n", model.schema_version.major, model.schema_version.minor)
	fmt.printf("Torch version: %s\n", model.torch_version)
	
	// Print graph info
	print_graph_summary(&model.graph_module.graph)
	
	// Create executor
	executor := create_executor_from_model(&model, weights)
	defer {
		for _, t in executor.tensors do delete(t)
		delete(executor.tensors)
		for &layer in executor.layers do layer_destroy(&layer)
		delete(executor.layers)
	}
	
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
