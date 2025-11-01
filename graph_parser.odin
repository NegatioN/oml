package main

import "core:fmt"
import "core:os"
import "core:encoding/json"

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
	inputs: []Node_Arg,
	outputs: []Node_Arg,
	nodes: []Graph_Node,
	tensor_values: map[string]Tensor_Meta,
}

// Unified argument structure that can hold different types
// JSON will populate whichever field is present
Node_Arg :: struct {
	as_tensor: Maybe(Tensor_Name),
	as_int:    Maybe(int),
	as_float:  Maybe(f32),
	as_bool:   Maybe(bool),
	as_string: Maybe(string),
}

Tensor_Name :: struct {
    name: string,
}

// Generate a unique name for a Node_Arg based on its content
// For scalars: "as_int_5", "as_float_3.14"
// For tensors: returns the tensor name directly
node_arg_to_name :: proc(arg: Node_Arg, allocator := context.allocator) -> (name: string, ok: bool) {
	context.allocator = allocator
	
	// If it's a tensor reference, return the tensor name
	if tensor_name, has_tensor := arg.as_tensor.?; has_tensor {
		return tensor_name.name, true
	}
	
	// If it's an integer, generate "as_int_VALUE"
	if int_val, has_int := arg.as_int.?; has_int {
		return fmt.tprintf("as_int_%d", int_val), true
	}
	
	// If it's a float, generate "as_float_VALUE"
	if float_val, has_float := arg.as_float.?; has_float {
		return fmt.tprintf("as_float_%v", float_val), true
	}
	
	// If it's a bool, generate "as_bool_true" or "as_bool_false"
	if bool_val, has_bool := arg.as_bool.?; has_bool {
		return fmt.tprintf("as_bool_%v", bool_val), true
	}
	
	// If it's a string, generate "as_string_VALUE"
	if str_val, has_string := arg.as_string.?; has_string {
		return fmt.tprintf("as_string_%s", str_val), true
	}
	
	// Unknown type
	return "", false
}

// Helper to get tensor data from a Node_Arg
// Returns the tensor from executor if it's a tensor reference,
// or creates a single-element array for scalar values
get_tensor_from_arg :: proc(arg: Node_Arg, executor: ^Graph_Executor, allocator := context.allocator) -> (tensor: []f32, ok: bool) {
    fmt.println(executor.constants)
    fmt.println(arg)
	context.allocator = allocator
	
	// Check if it's a tensor reference

	if tensor_name, has_tensor := node_arg_to_name(arg); has_tensor {
		// Look up tensor in executor's runtime tensors, constants, or weights
		if t, found := executor.tensors[tensor_name]; found {
			return t, true
		}
		if c, found := executor.constants[tensor_name]; found {
			return c, true
		}
		if w, found := executor.weights[tensor_name]; found {
			return w, true
		}
	}
	return nil, false
}

Node_Operation :: enum {
    Linear, ReLU, Add, Arange, Unknown,
}

parse_operation :: proc(target: string) -> Node_Operation {
    /*
    Other relevant operators:
    torch.ops.aten.to.dtype

    */
    switch target {
    case "torch.ops.aten.linear.default":
        return .Linear
    case "torch.ops.aten.relu.default":
        return .ReLU
    case "torch.ops.aten.add.Tensor":
        return .Add
    case "torch.ops.aten.arange.default":
        return .Arange
    case:
        return .Unknown
    }
}

Graph_Node :: struct {
	target: string,  // e.g., "torch.ops.aten.linear.default"
    operation: Node_Operation,
	inputs: []Node_Input,
	outputs: []Node_Arg,
	metadata: Maybe(Node_Metadata),
}

Node_Input :: struct {
	name: string,
	arg: Node_Arg,
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
	tensor_constant: Maybe(Tensor_Constant_Info),
	user_input: Maybe(User_Input_Info),
}

Parameter_Info :: struct {
	arg: Tensor_Name,
	parameter_name: string,
}

Tensor_Constant_Info :: struct {
	arg: Tensor_Name,
	tensor_constant_name: string,
}

User_Input_Info :: struct {
	arg: Node_Arg,
}

Output_Spec :: struct {
	user_output: User_Output_Info,
}

User_Output_Info :: struct {
	arg: Node_Arg,
}

Schema_Version :: struct {
	major: int,
	minor: int,
}

// Graph execution context
Graph_Executor :: struct {
	graph: ^Graph,
	weights: map[string][]f32,       // Pre-loaded model weights
	constants: map[string][]f32,     // Pre-loaded constant tensors (from files + inline scalars)
	tensors: map[string][]f32,       // Runtime tensor values
	layers: []Layer,                 // Pre-built layers in execution order
}

// Build constants map from file-based constants and inline scalars in the graph
build_constants_from_graph :: proc(graph: ^Graph, model_dir: string) -> (constants: map[string][]f32, ok: bool) {
	constants = make(map[string][]f32)
	
	// 1. Load file-based constants from model/data/constants/
	constants_config_path := fmt.tprintf("%s/data/constants/model_constants_config.json", model_dir)
	if config_data, read_ok := os.read_entire_file(constants_config_path); read_ok {
		defer delete(config_data)
		
		// Parse constants config (similar structure to weights config)
		Constants_Config_Root :: struct {
			config: map[string]Weight_Config,
		}
		
		Weight_Config :: struct {
			path_name:   string,
			is_param:    bool,
			use_pickle:  bool,
			tensor_meta: Tensor_Meta,
		}
		
		config: Constants_Config_Root
		parse_err := json.unmarshal(config_data, &config)
		if parse_err == nil {
			defer delete(config.config)
			
			// Load each constant file
			for const_name, const_info in config.config {
				const_path := fmt.tprintf("%s/data/constants/%s", model_dir, const_info.path_name)
				const_data, c_ok := os.read_entire_file(const_path)
				if c_ok {
					defer delete(const_data)
					
					// Convert bytes to floats
					num_floats := len(const_data) / size_of(f32)
					const_floats := transmute([]f32)const_data
					
					constants[const_name] = make([]f32, num_floats)
					copy(constants[const_name], const_floats[:num_floats])
					
					fmt.printf("Loaded constant %s: %v floats\n", const_name, num_floats)
				}
			}
		}
	}
	
	// 2. Extract inline scalar constants from graph nodes
	for &node, node_idx in graph.nodes {
		for &input, input_idx in node.inputs {
			// Generate a name for this argument if it's a scalar
			arg_name, name_ok := node_arg_to_name(input.arg)
			if !name_ok do continue
			
			// Skip if it's a tensor reference (already handled elsewhere)
			if input.arg.as_tensor != nil do continue
			
			// Extract scalar value
			scalar_value: Maybe(f32)
			
			if int_val, has_int := input.arg.as_int.?; has_int {
				scalar_value = f32(int_val)
			} else if float_val, has_float := input.arg.as_float.?; has_float {
				scalar_value = float_val
			}
			
			// If we found a scalar, store it with the generated name
			if val, has_scalar := scalar_value.?; has_scalar {
				// Only add if not already present (avoid duplicates)
				if arg_name not_in constants {
					scalar_array := make([]f32, 1)
					scalar_array[0] = val
					constants[arg_name] = scalar_array
				}
			}
		}
	}
	
	return constants, true
}

// Build layers from graph nodes during initialization
build_layers_from_graph :: proc(graph: ^Graph, weights: map[string][]f32) -> []Layer {
	layers := make([dynamic]Layer)
	
	for &node in graph.nodes {
		layer: Layer
		
		switch node.operation {
		case Node_Operation.Linear:
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
			layer = linear_create(in_features, out_features, weight, bias)

		case Node_Operation.ReLU:
			layer := relu_create()

        case Node_Operation.Add:
            //weight_name := node.inputs[1].arg.as_tensor.?.name  //TODO if we add the value manually here as an int, the fetching is different.
            layer = add_create()

        case Node_Operation.Arange:
            end := node.inputs[0].arg.as_int.?
            layer = arange_create(end)
		case Node_Operation.Unknown:
            fmt.printf("Unsupported operation: %s\n", node.target)
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

    //TODO we cant assume that the input is called x forever
	// Set input tensor (assuming single input named 'x')
	executor.tensors["x"] = make([]f32, len(input))
	copy(executor.tensors["x"], input)

    it := soa_zip(node=executor.graph.nodes, layer=executor.layers)
	// Execute each node and its corresponding layer in order
    //TODO the graph might not be so easy to execute in order when the model is more complex?
    for &el, i  in it {
        //fmt.printf("%v %v\n\n", i, &el.node) //Debuglog
		execute_node(executor, &el.node, &el.layer) or_return
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
	switch node.operation {
	case Node_Operation.Linear:
		return execute_linear_node(executor, node, layer)
	case Node_Operation.ReLU:
		return execute_relu_node(executor, node, layer)
    case Node_Operation.Add:
        return execute_add_node(executor, node, layer)
    case Node_Operation.Arange:
        return execute_arange_node(executor, node, layer)
	case Node_Operation.Unknown:
		fmt.printf("Unsupported operation: %s\n", node.target)
		return false
	}
    return true // :thinking:
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
	// Get input and output names
	input_name := node.inputs[0].arg.as_tensor.?.name
	output_name := node.outputs[0].as_tensor.?.name

	// Fetch input tensor
	input := executor.tensors[input_name]

	// Allocate output
	output := make([]f32, len(input))

	// Execute the layer
	relu_forward(input, output)

	// Store result
	executor.tensors[output_name] = output
	return true
}

execute_add_node :: proc(executor: ^Graph_Executor, node: ^Graph_Node, layer: ^Layer) -> bool {
	// Get input tensors using helper function
	input_1, ok1 := get_tensor_from_arg(node.inputs[0].arg, executor)
	if !ok1 {
		fmt.println("Failed to get first input for add operation")
		return false
	}

	input_2,  ok2 := get_tensor_from_arg(node.inputs[1].arg, executor)
	if !ok2 {
		fmt.println("Failed to get second input for add operation")
		return false
	}

	// Determine output size (larger of the two inputs)
	output_size := max(len(input_1), len(input_2))
	output := make([]f32, output_size)
	
	// Execute add operation
	add_forward(input_1, input_2, output)
	
	// Store result
	output_name := node.outputs[0].as_tensor.?.name
	executor.tensors[output_name] = output
	
	return true
}

execute_arange_node:: proc(executor: ^Graph_Executor, node: ^Graph_Node, layer: ^Layer) -> bool {
    arange_layer := layer.(Arange) or_return

    // Get input and output names
    output_name := node.outputs[0].as_tensor.?.name

    // Allocate output
    output := make([]f32, arange_layer.end)

    for i in 0..<len(output){
        output[i] = cast(f32)i
    }
    //arange_forward(output)

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
