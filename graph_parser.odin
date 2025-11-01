package main

import "core:fmt"

// PyTorch Graph structures for parsing model.json

Graph_Model :: struct {
	graph_module: Graph_Module,
	opset_version: map[string]int,
	schema_version: Schema_Version,
	torch_version: string,
}

Graph_Module :: struct {
	graph: Graph,
	signature: Signature,
}

Graph :: struct {
	inputs: []Tensor_Arg,
	outputs: []Tensor_Arg,
	nodes: []Graph_Node,
	tensor_values: map[string]Tensor_Meta,
}

Tensor_Arg :: struct {
	as_tensor: Maybe(Tensor_Name),
}

Tensor_Name :: struct {
	name: string,
}

Graph_Node :: struct {
	target: string,  // e.g., "torch.ops.aten.linear.default"
	inputs: []Node_Input,
	outputs: []Tensor_Arg,
	metadata: Maybe(Node_Metadata),
}

Node_Input :: struct {
	name: string,
	arg: Tensor_Arg,
	kind: int,
}

Node_Metadata :: struct {
	stack_trace: string,
	nn_module_stack: string,
	torch_fn: string,
}

Tensor_Meta :: struct {
	dtype: int,  // 7 = float32
	sizes: []Size_Entry,
	requires_grad: bool,
	device: Device_Info,
	strides: []Size_Entry,
	storage_offset: Size_Entry,
	layout: int,
}

Size_Entry :: struct {
	as_int: int,
}

Device_Info :: struct {
	type: string,
	index: Maybe(int),
}

Signature :: struct {
	input_specs: []Input_Spec,
	output_specs: []Output_Spec,
}

Input_Spec :: struct {
	parameter: Maybe(Parameter_Info),
	user_input: Maybe(User_Input_Info),
}

Parameter_Info :: struct {
	arg: Tensor_Name,
	parameter_name: string,
}

User_Input_Info :: struct {
	arg: Tensor_Arg,
}

Output_Spec :: struct {
	user_output: User_Output_Info,
}

User_Output_Info :: struct {
	arg: Tensor_Arg,
}

Schema_Version :: struct {
	major: int,
	minor: int,
}

// Graph execution context
Graph_Executor :: struct {
	graph: ^Graph,
	weights: map[string][]f32,
	tensors: map[string][]f32,  // Runtime tensor values
	layers: []Layer,             // Pre-built layers in execution order
}

// Build layers from graph nodes during initialization
build_layers_from_graph :: proc(graph: ^Graph, weights: map[string][]f32) -> []Layer {
	layers := make([dynamic]Layer)
	
	for &node in graph.nodes {
		layer: Layer
		
		switch node.target {
		case "torch.ops.aten.linear.default":
			// Extract layer parameters
			weight_name := node.inputs[1].arg.as_tensor.?.name
			bias_name := node.inputs[2].arg.as_tensor.?.name
			
			weight := weights[weight_name]
			bias := weights[bias_name]
			
			// Get dimensions from metadata
			weight_meta := graph.tensor_values[weight_name]
			out_features := weight_meta.sizes[0].as_int
			in_features := weight_meta.sizes[1].as_int
			
			// Create Linear layer
			linear_layer := linear_create(in_features, out_features, weight, bias)
			layer = linear_layer
			
		case "torch.ops.aten.relu.default":
			// Create ReLU layer (stateless)
			relu_layer := relu_create()
			layer = relu_layer
			
		case:
			// Unknown layer type - could log warning
			continue
		}
		
		append(&layers, layer)
	}
	
	return layers[:]
}

// Execute the graph with input
execute_graph :: proc(executor: ^Graph_Executor, input: []f32) -> (output: []f32, ok: bool) {
	// Clear intermediate tensors from previous runs
	for key in executor.tensors {
		delete(executor.tensors[key])
	}
	clear(&executor.tensors)
	
	// Set input tensor (assuming single input named 'x')
	executor.tensors["x"] = make([]f32, len(input))
	copy(executor.tensors["x"], input)
	
	// Execute each node and its corresponding layer in order
	for &node, i in executor.graph.nodes {
		execute_node(executor, &node, &executor.layers[i]) or_return
	}
	
	// Get output tensor (assuming single output)
	if len(executor.graph.outputs) == 0 {
		return nil, false
	}
	
	output_name := executor.graph.outputs[0].as_tensor.?.name
	output_tensor, found := executor.tensors[output_name]
	if !found {
		fmt.printf("Output tensor '%s' not found\n", output_name)
		return nil, false
	}
	
	// Return copy of output
	result := make([]f32, len(output_tensor))
	copy(result, output_tensor)
	return result, true
}

// Execute a single node with its pre-built layer
execute_node :: proc(executor: ^Graph_Executor, node: ^Graph_Node, layer: ^Layer) -> bool {
	switch node.target {
	case "torch.ops.aten.linear.default":
		return execute_linear_node(executor, node, layer)
	case "torch.ops.aten.relu.default":
		return execute_relu_node(executor, node, layer)
	case:
		fmt.printf("Unsupported operation: %s\n", node.target)
		return false
	}
}

// Execute linear node using pre-built Linear layer
execute_linear_node :: proc(executor: ^Graph_Executor, node: ^Graph_Node, layer: ^Layer) -> bool {
	linear_layer := layer.(Linear) or_return
	
	// Get input and output names
	input_name := node.inputs[0].arg.as_tensor.?.name
	output_name := node.outputs[0].as_tensor.?.name
	
	// Fetch input tensor
	input := executor.tensors[input_name] if input_name in executor.tensors else executor.weights[input_name]
	
	// Allocate output
	output := make([]f32, linear_layer.out_features)
	
	// Execute the layer
	linear_forward(&linear_layer, input, output)
	
	// Store result
	executor.tensors[output_name] = output
	return true
}

// Execute ReLU node using pre-built ReLU layer
execute_relu_node :: proc(executor: ^Graph_Executor, node: ^Graph_Node, layer: ^Layer) -> bool {
	relu_layer := layer.(ReLU) or_return
	
	// Get input and output names
	input_name := node.inputs[0].arg.as_tensor.?.name
	output_name := node.outputs[0].as_tensor.?.name
	
	// Fetch input tensor
	input := executor.tensors[input_name]
	
	// Allocate output
	output := make([]f32, len(input))
	
	// Execute the layer
	relu_forward(&relu_layer, input, output)
	
	// Store result
	executor.tensors[output_name] = output
	return true
}

// Print graph summary
print_graph_summary :: proc(graph: ^Graph) {
	fmt.println("\n=== Graph Summary ===")
	
	fmt.println("\nInputs:")
	for input in graph.inputs {
		if tensor, ok := input.as_tensor.?; ok {
			fmt.printf("  - %s\n", tensor.name)
		}
	}
	
	fmt.println("\nOutputs:")
	for output in graph.outputs {
		if tensor, ok := output.as_tensor.?; ok {
			fmt.printf("  - %s\n", tensor.name)
		}
	}
	
	fmt.println("\nNodes:")
	for node, i in graph.nodes {
		fmt.printf("\n%d. %s\n", i+1, node.target)
		fmt.println("   Inputs:")
		for inp in node.inputs {
			if tensor, ok := inp.arg.as_tensor.?; ok {
				fmt.printf("     - %s: %s\n", inp.name, tensor.name)
			}
		}
		fmt.println("   Outputs:")
		for out in node.outputs {
			if tensor, ok := out.as_tensor.?; ok {
				fmt.printf("     - %s\n", tensor.name)
			}
		}
	}
	
	fmt.println("\nTensor Metadata:")
	for name, meta in graph.tensor_values {
		sizes := make([dynamic]int)
		defer delete(sizes)
		for size in meta.sizes {
			append(&sizes, size.as_int)
		}
		fmt.printf("  %s: shape=%v, dtype=%d\n", name, sizes[:], meta.dtype)
	}
}
