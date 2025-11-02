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
    as_tensors: Maybe([]Tensor_Name), // For now we remap this to single arguments
	as_int:    Maybe(int),
	as_float:  Maybe(f32),
	as_bool:   Maybe(bool),
	as_string: Maybe(string),
}

Tensor_Name :: struct {
    name: string,
}

// Expand a Node_Arg with as_tensors into multiple Node_Args with as_tensor
// This normalizes variadic inputs into a flat list of single-tensor arguments
expand_node_inputs :: proc(inputs: []Node_Input, allocator := context.allocator) -> []Node_Input {
	context.allocator = allocator
	
	expanded := make([dynamic]Node_Input)
	
	for input in inputs {
		// If it has multiple tensors, expand them into separate Node_Inputs
		if tensors, ok := input.arg.as_tensors.?; ok {
			for tensor in tensors {
				expanded_input := Node_Input{
					name = input.name,
					kind = input.kind,
					arg = Node_Arg{
						as_tensor = tensor,
					},
				}
				append(&expanded, expanded_input)
			}
		} else {
			// Keep as-is (single tensor or scalar)
			append(&expanded, input)
		}
	}
	
	return expanded[:]
}

// Expand Node_Arg outputs with as_tensors into multiple as_tensor outputs
expand_node_outputs :: proc(outputs: []Node_Arg, allocator := context.allocator) -> []Node_Arg {
	context.allocator = allocator
	
	expanded := make([dynamic]Node_Arg)
	
	for output in outputs {
		// If it has multiple tensors, expand them
		if tensors, ok := output.as_tensors.?; ok {
			for tensor in tensors {
				expanded_output := Node_Arg{
					as_tensor = tensor,
				}
				append(&expanded, expanded_output)
			}
		} else {
			// Keep as-is
			append(&expanded, output)
		}
	}
	
	return expanded[:]
}

// TODO: Handle variadic operations properly
// Some operations like concat, stack, etc. need to know which inputs belong together.
// Consider adding a variadic_group_id field or keeping track of original structure.
// For now, expand_node_inputs/outputs flattens everything, which works for simple cases
// but will need refinement for operations that semantically require grouped inputs.

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

// Helper to get tensor data from a Node_Arg even if its a weight, tensor or constant
get_tensor_from_arg :: proc(arg: Node_Arg, executor: ^Graph_Executor, allocator := context.allocator) -> (tensor: Tensor, ok: bool) {
	context.allocator = allocator
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
	return {}, false
}

Node_Operation :: enum {
    Linear, ReLU, Add, Sub, Arange, LiftFreshCopy, Cat, Noop, LayerNorm, Unknown,
}

parse_operation :: proc(target: string) -> Node_Operation {
    /*
    Other relevant operators:
    torch.ops.aten.to.dtype
    */
    switch target {
    case "torch.ops.aten.linear.default":
        return .Linear
    case "torch.ops.aten.layer_norm.default":
        return .LayerNorm
    case "torch.ops.aten.relu.default":
        return .ReLU
    case "torch.ops.aten.gelu.default": // Circle back and implement GELU
        return .ReLU
    case "torch.ops.aten.add.Tensor":
        return .Add
    case "torch.ops.aten.sub.Tensor":
        return .Sub
    case "torch.ops.aten.arange.default":
        return .Arange
    case "torch.ops.aten.lift_fresh_copy.default":
        return .LiftFreshCopy
    case "torch.ops.aten.dropout.default":
        return .Noop
    case "torch.ops.aten.detach_.default":
        return .Noop
    case "torch.ops.aten.cat.default":
        return .Cat
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

// Config for loading weights/constants from files (shared structure)
Weight_Config :: struct {
	path_name:   string,
	is_param:    bool,
	use_pickle:  bool,
	tensor_meta: Tensor_Meta,
}

// Helper: Load binary f32 tensor from file
load_f32_tensor_from_file :: proc(path: string) -> (tensor: []f32, ok: bool) {
	data, read_ok := os.read_entire_file(path)
	if !read_ok do return nil, false
	defer delete(data)
	
	num_floats := len(data) / size_of(f32)
	floats := transmute([]f32)data
	
	result := make([]f32, num_floats)
	copy(result, floats[:num_floats])
	return result, true
}

// Helper: Load tensor config from JSON file and load all tensor files as Tensors
load_tensor_config :: proc(config_path: string, data_dir: string) -> (tensors: map[string]Tensor, ok: bool) {
	tensors = make(map[string]Tensor)
	
	config_data, read_ok := os.read_entire_file(config_path)
	if !read_ok do return tensors, false
	defer delete(config_data)
	
	// Parse config
	Config_Root :: struct {
		config: map[string]Weight_Config,
	}
	
	config: Config_Root
	parse_err := json.unmarshal(config_data, &config)
	if parse_err != nil do return tensors, false
	defer delete(config.config)
	
	// Load each tensor file and create Tensor from metadata
	for tensor_name, tensor_info in config.config {
		tensor_path := fmt.tprintf("%s/%s", data_dir, tensor_info.path_name)
		tensor_data, t_ok := load_f32_tensor_from_file(tensor_path)
		if t_ok {
			// Create Tensor with metadata
			tensor := meta_data_to_tensor(tensor_data, tensor_info.tensor_meta)
			tensors[tensor_name] = tensor
			fmt.printf("Loaded %s: %d floats from %s (shape: %v)\n", 
				tensor_name, len(tensor_data), tensor_info.path_name, tensor.sizes)
		}
	}
	
	return tensors, true
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
	weights: map[string]Tensor,      // Pre-loaded model weights as tensors
	constants: map[string]Tensor,    // Pre-loaded constant tensors (from files + inline scalars)
	tensors: map[string]Tensor,      // Runtime tensor values
	layers: []Layer,                 // Pre-built layers in execution order
}

// Build constants map from file-based constants and inline scalars in the graph
build_constants_from_graph :: proc(graph: ^Graph, model_dir: string) -> (constants: map[string]Tensor, ok: bool) {
	constants = make(map[string]Tensor)
	
	// 1. Load file-based constants from model/data/constants/ using helper
	constants_config_path := fmt.tprintf("%s/data/constants/model_constants_config.json", model_dir)
	file_constants, loaded := load_tensor_config(constants_config_path, fmt.tprintf("%s/data/constants", model_dir))
	if loaded {
		// Merge file constants into our map
		for name, tensor in file_constants {
			constants[name] = tensor
		}
		delete(file_constants)
	}
	
	// 2. Extract inline scalar constants from graph nodes
	for &node, node_idx in graph.nodes {
		for &input, input_idx in node.inputs {
			// Generate a name for this argument if it's a scalar
			arg_name, name_ok := node_arg_to_name(input.arg)
			if !name_ok do continue
			
			// Skip if it's a tensor reference (already handled elsewhere)
			if input.arg.as_tensor != nil || input.arg.as_tensors != nil do continue
			
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
					// Create a scalar tensor (0-D or 1-D with single element)
					scalar_data := make([]f32, 1)
					scalar_data[0] = val
					scalar_tensor := Tensor{
						data = scalar_data,
						sizes = []int{1},      // Scalar as 1-element tensor
						strides = []int{1},
					}
					constants[arg_name] = scalar_tensor
				}
			}
		}
	}
	
	return constants, true
}

// Build layers from graph nodes during initialization
build_layers_from_graph :: proc(graph: ^Graph, weights: map[string]Tensor) -> []Layer {
	layers := make([dynamic]Layer)
	
	for &node in graph.nodes {
		layer: Layer
		
		switch node.operation {
		case Node_Operation.Linear:
			// Extract layer parameters
			weight_name := node.inputs[1].arg.as_tensor.?.name
			bias_name := node.inputs[2].arg.as_tensor.?.name
			
			weight_tensor := weights[weight_name]
			bias_tensor := weights[bias_name]
			
			// Get dimensions from tensor sizes
			out_features := weight_tensor.sizes[0]
			in_features := weight_tensor.sizes[1]
			
			// Create Linear layer with tensors
			layer = linear_create(in_features, out_features, weight_tensor, bias_tensor)


        case Node_Operation.LayerNorm:
            fmt.println(node)
        case Node_Operation.Arange:
            end := node.inputs[0].arg.as_int.?
            layer = arange_create(end)

        case Node_Operation.ReLU:
        case Node_Operation.Add:
        case Node_Operation.Sub:
        case Node_Operation.LiftFreshCopy:
        case Node_Operation.Noop:
        case Node_Operation.Cat:
            // Find the 'dim' parameter in the node inputs
            dim := 0  // Default value if not found
            for input in node.inputs {
                if input.name == "dim" {
                    if dim_val, ok := input.arg.as_int.?; ok {
                        dim = dim_val
                    }
                    break
                }
            }
            layer = cat_create(dim)
		case Node_Operation.Unknown:
            fmt.printf("Unsupported operation: %s\n", node.target)
			// Unknown layer type - could log warning
			continue
		}

        //For now, always append an empty layer even if it doesnt do anything. We rely on Nodes and Layers being identical length and in order.
		append(&layers, layer)
	}
	
	return layers[:]
}

// Execute the graph with input
execute_graph :: proc(executor: ^Graph_Executor, input: []f32) -> (output: []f32, ok: bool) {
	// Clear intermediate tensors from previous runs
	for _, tensor in executor.tensors {
		destroy_tensor(tensor)
	}
	clear(&executor.tensors)

    //TODO we cant assume that the input is called x forever
	// Set input tensor (assuming single input named 'x')
	input_data := make([]f32, len(input))
	copy(input_data, input)
	executor.tensors["x"] = make_1d_tensor(input_data)

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
	
	// Return copy of output data
	result := make([]f32, len(output_tensor.data))
	copy(result, output_tensor.data)
	return result, true
}


// Execute a single node with its pre-built layer
execute_node :: proc(executor: ^Graph_Executor, node: ^Graph_Node, layer: ^Layer) -> bool {
    //TODO Nodes could contain their own function or something. seems like a useless switch statement.
	switch node.operation {
	case Node_Operation.Linear:
		return execute_linear_node(executor, node, layer)
    case Node_Operation.LayerNorm:
        return execute_linear_node(executor, node, layer)
	case Node_Operation.ReLU:
		return execute_relu_node(executor, node)
    case Node_Operation.Add:
        return execute_add_node(executor, node)
    case Node_Operation.Sub:
        return execute_sub_node(executor, node)
    case Node_Operation.Arange:
        return execute_arange_node(executor, node, layer)
    case Node_Operation.LiftFreshCopy:
        return execute_copy_node(executor, node)
    case Node_Operation.Noop:
        return execute_copy_node(executor, node)
    case Node_Operation.Cat:
        return execute_cat_node(executor, node, layer)
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
	input_tensor := executor.tensors[input_name] if input_name in executor.tensors else executor.weights[input_name]
	
	// Create output tensor
	output_data := make([]f32, linear_layer.out_features)
	output_tensor := make_1d_tensor(output_data)
	
	// Execute the layer
	linear_forward(&linear_layer, input_tensor, &output_tensor)
	
	// Store result
	executor.tensors[output_name] = output_tensor
	return true
}

// Execute ReLU node using pre-built ReLU layer
execute_relu_node :: proc(executor: ^Graph_Executor, node: ^Graph_Node) -> bool {
	// Get input and output names
	input_name := node.inputs[0].arg.as_tensor.?.name
	output_name := node.outputs[0].as_tensor.?.name

	// Fetch input tensor
	input_tensor := executor.tensors[input_name]

	// Create output tensor
	output_data := make([]f32, len(input_tensor.data))
	output_tensor := make_1d_tensor(output_data)

	// Execute the layer
	relu_forward(input_tensor, &output_tensor)

	// Store result
	executor.tensors[output_name] = output_tensor
	return true
}

execute_add_node :: proc(executor: ^Graph_Executor, node: ^Graph_Node) -> bool {
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
	output_size := max(len(input_1.data), len(input_2.data))
	output_data := make([]f32, output_size)
	output_tensor := make_1d_tensor(output_data)
	
	// Execute add operation
	add_tensors(input_1, input_2, &output_tensor)
	
	// Store result
	output_name := node.outputs[0].as_tensor.?.name
	executor.tensors[output_name] = output_tensor

	return true
}


execute_sub_node :: proc(executor: ^Graph_Executor, node: ^Graph_Node) -> bool {
    input_1, ok1 := get_tensor_from_arg(node.inputs[0].arg, executor)
    if !ok1 {
        fmt.println("Failed to get first input for sub operation")
        return false
    }

    input_2,  ok2 := get_tensor_from_arg(node.inputs[1].arg, executor)
    if !ok2 {
        fmt.println("Failed to get second input for sub operation")
        return false
    }

    // Determine output size (larger of the two inputs)
    output_size := max(len(input_1.data), len(input_2.data))
    output_data := make([]f32, output_size)
	output_tensor := make_1d_tensor(output_data)

    // Execute sub operation
    sub_tensors(input_1, input_2, &output_tensor)

    // Store result
    output_name := node.outputs[0].as_tensor.?.name
    executor.tensors[output_name] = output_tensor

    	return true
}

execute_arange_node:: proc(executor: ^Graph_Executor, node: ^Graph_Node, layer: ^Layer) -> bool {
    arange_layer := layer.(Arange) or_return

    // Get output name
    output_name := node.outputs[0].as_tensor.?.name

    // Allocate output
    output_data := make([]f32, arange_layer.end)

    for i in 0..<len(output_data){
        output_data[i] = cast(f32)i
    }

    // Store result
    executor.tensors[output_name] = make_1d_tensor(output_data)
    return true
}

execute_cat_node:: proc(executor: ^Graph_Executor, node: ^Graph_Node, layer: ^Layer) -> bool {
    l := layer.(Cat) or_return
    input_1, ok1 := get_tensor_from_arg(node.inputs[0].arg, executor)
    if !ok1 {
        fmt.println("Failed to get first input for cat")
        return false
    }
    input_2, ok2 := get_tensor_from_arg(node.inputs[1].arg, executor)
    if !ok2 {
        fmt.println("Failed to get second input for cat")
        return false
    }
    
    output_name := node.outputs[0].as_tensor.?.name
    output_data := make([]f32, len(input_1.data)+len(input_2.data))
    
    if l.dim == 0 {
        copy(output_data[:len(input_1.data)], input_1.data)
        copy(output_data[len(input_1.data):], input_2.data)
    }
    else {
        panic("Unsupported dimension for catting")
    }

    executor.tensors[output_name] = make_1d_tensor(output_data)
    return true
}

execute_copy_node:: proc(executor: ^Graph_Executor, node: ^Graph_Node) -> bool {
    input_1, ok1 := get_tensor_from_arg(node.inputs[0].arg, executor)
    if !ok1 {
        fmt.println("Failed to get first input for copy")
        return false
    }
    output_name := node.outputs[0].as_tensor.?.name

    output_data := make([]f32, len(input_1.data))
    copy(output_data, input_1.data)

    executor.tensors[output_name] = make_1d_tensor(output_data)
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
