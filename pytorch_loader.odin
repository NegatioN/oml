package main

/*
import "core:fmt"
import "core:os"
import "core:encoding/json"
import "core:mem"
import "core:strings"

// PyTorch model weight configuration
Weight_Config :: struct {
	path_name:    string,
	is_param:     bool,
	use_pickle:   bool,
	tensor_meta:  Tensor_Meta,
}

Tensor_Meta :: struct {
	dtype:          int,
	sizes:          []Size_Entry,
	requires_grad:  bool,
	device:         Device_Info,
	strides:        []Size_Entry,
	storage_offset: Size_Entry,
	layout:         int,
}

Size_Entry :: struct {
	as_int: int,
}

Device_Info :: struct {
	type:  string,
	index: json.Null,
}

Weights_Config :: struct {
	config: map[string]Weight_Config,
}

// Load a PyTorch .pt file (must be unzipped first)
load_pytorch_model :: proc(model_dir: string) -> (weights: map[string][]f32, ok: bool) {
	weights = make(map[string][]f32)
	
	// Read weights config JSON
	config_path := fmt.tprintf("%s/data/weights/model_weights_config.json", model_dir)
	config_data, read_ok := os.read_entire_file(config_path)
	if !read_ok {
		fmt.println("Failed to read config:", config_path)
		return weights, false
	}
	defer delete(config_data)
	
	// Parse JSON (simplified - you'd need proper JSON parsing here)
	// For now, we'll manually load the weights knowing the structure
	
	// Load weight_0 (fc1.weight)
	weight_0_path := fmt.tprintf("%s/data/weights/weight_0", model_dir)
	weight_0_data, w0_ok := os.read_entire_file(weight_0_path)
	if !w0_ok {
		fmt.println("Failed to read weight_0")
		return weights, false
	}
	defer delete(weight_0_data)
	
	// Convert bytes to f32 (assuming little-endian float32)
    fmt.printf("  %s\n", weight_0_data)
	weight_0_floats := transmute([]f32)weight_0_data
	weights["fc1.weight"] = make([]f32, len(weight_0_floats))
	copy(weights["fc1.weight"], weight_0_floats)
	
	// Load weight_1 (fc1.bias)
	weight_1_path := fmt.tprintf("%s/data/weights/weight_1", model_dir)
	weight_1_data, w1_ok := os.read_entire_file(weight_1_path)
	if !w1_ok {
		fmt.println("Failed to read weight_1")
		return weights, false
	}
	defer delete(weight_1_data)

    fmt.printf("  %v\n", len(weight_1_data))
	weight_1_floats := transmute([]f32)weight_1_data
	weights["fc1.bias"] = make([]f32, len(weight_1_floats))
	copy(weights["fc1.bias"], weight_1_floats)
	
	return weights, true
}

// Helper to print loaded weights
print_weights :: proc(weights: map[string][]f32) {
	fmt.println("Loaded weights:")
	for name, values in weights {
		fmt.printf("  %s: %v\n", name, values)
	}
}

*/