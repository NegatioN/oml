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
	
	// Execute each node in order
	for &node in executor.graph.nodes {
		switch node.target {
		case "torch.ops.aten.linear.default":
			execute_linear(executor, &node) or_return
		case "torch.ops.aten.relu.default":
			execute_relu(executor, &node) or_return
		case:
			fmt.printf("Unsupported operation: %s\n", node.target)
			return nil, false
		}
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

// Execute linear operation: output = input @ weight^T + bias
execute_linear :: proc(executor: ^Graph_Executor, node: ^Graph_Node) -> bool {
	// Get input tensors by name
	input_name := node.inputs[0].arg.as_tensor.?.name
	weight_name := node.inputs[1].arg.as_tensor.?.name
	bias_name := node.inputs[2].arg.as_tensor.?.name
	output_name := node.outputs[0].as_tensor.?.name
	
	// Fetch tensors
	input := executor.tensors[input_name] if input_name in executor.tensors else executor.weights[input_name]
	weight := executor.weights[weight_name]
	bias := executor.weights[bias_name]
	
	// Debug: Print what we're using
	fmt.printf("[DEBUG] Linear: input=%s(%v), weight=%s(%v), bias=%s(%v)\n", 
		input_name, input, weight_name, weight, bias_name, bias)
	
	// Get shapes from metadata
	weight_meta := executor.graph.tensor_values[weight_name]
	out_features := weight_meta.sizes[0].as_int
	in_features := weight_meta.sizes[1].as_int
	
	fmt.printf("[DEBUG] Linear shapes: out_features=%d, in_features=%d\n", out_features, in_features)
	
	// Allocate output
	output := make([]f32, out_features)
	
	// Compute: output = input @ weight^T + bias
	for i in 0..<out_features {
		sum := bias[i]
		for j in 0..<in_features {
			weight_idx := i * in_features + j
			sum += input[j] * weight[weight_idx]
		}
		output[i] = sum
	}
	
	fmt.printf("[DEBUG] Linear output: %v\n", output)
	
	executor.tensors[output_name] = output
	return true
}

// Execute ReLU operation: output = max(0, input)
execute_relu :: proc(executor: ^Graph_Executor, node: ^Graph_Node) -> bool {
	input_name := node.inputs[0].arg.as_tensor.?.name
	output_name := node.outputs[0].as_tensor.?.name
	
	input := executor.tensors[input_name]
	output := make([]f32, len(input))
	
	for i in 0..<len(input) {
		output[i] = max(0, input[i])
	}
	
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
