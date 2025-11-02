package main

Tensor :: struct {
    data: []f32, //TODO datatypes
    sizes: []int, //TODO find all
    strides: []int,
    //assume storage_offset is always 0 to simplify
}

meta_data_to_tensor :: proc(data: []f32, meta:Tensor_Meta) -> Tensor {
    sizes := make([]int, len(meta.sizes))
    for &e, i in meta.sizes {
        sizes[i] = e.as_int
    }
    strides:= make([]int, len(meta.strides))
    for &e, i in meta.strides{
        strides[i] = e.as_int
    }
    return Tensor{data=data, sizes = sizes, strides = strides}
}

destroy_tensor :: proc(t: Tensor) {
    delete(t.data)
    delete(t.strides)
    delete(t.sizes)
}

// Helper: create a simple 1D tensor from data
make_1d_tensor :: proc(data: []f32) -> Tensor {
    sizes := make([]int, 1)
    sizes[0] = len(data)

    strides := make([]int, 1)
    strides[0] = 1

    return Tensor{
        data = data,
        sizes = sizes,
        strides = strides,
    }
}

copy_tensor :: proc(t: Tensor) -> Tensor {
    data := make([]f32, len(t.data))
    copy(data, t.data)
    
    sizes := make([]int, len(t.sizes))
    copy(sizes, t.sizes)
    
    strides := make([]int, len(t.strides))
    copy(strides, t.strides)
    
    return Tensor{
        data = data,
        sizes = sizes,
        strides = strides,
    }
}

// this might not be needed right away
size_to_strides :: proc(sizes: []int) -> []int {
    // ex: (2, 3, 4)size would result in -> (12, 4, 1) strides
    strides := make([]int, len(sizes))
    strides[len(strides)-1] = 1 // last stride is always 1
    if len(sizes) == 1 {
    } else {
        for i in len(strides)-1..<0 {


        }
    }
    return strides
}

add_tensors :: proc(inp: Tensor, other: Tensor, output: ^Tensor) {
    broadcast_binary_op(inp, other, output, .Add)
}
sub_tensors :: proc(inp: Tensor, other: Tensor, output: ^Tensor) {
    broadcast_binary_op(inp, other, output, .Sub)
}

// Compute linear index from multi-dimensional indices using strides
compute_index :: proc(indices: []int, strides: []int) -> int {
    assert(len(indices) == len(strides))
    idx := 0
    for i in 0..<len(indices) {
        idx += indices[i] * strides[i]
    }
    return idx
}

// Check if two shapes are broadcast-compatible
// Returns the broadcast shape if compatible, nil otherwise
check_broadcast_shapes :: proc(shape_a, shape_b: []int, allocator := context.allocator) -> []int {
// Align shapes from the right (trailing dimensions)
    ndim_a := len(shape_a)
    ndim_b := len(shape_b)
    ndim_out := max(ndim_a, ndim_b)

    result := make([]int, ndim_out, allocator)

    // Work backwards from the last dimension
    for i in 0..<ndim_out {
        dim_a := i < ndim_a ? shape_a[ndim_a - 1 - i] : 1
        dim_b := i < ndim_b ? shape_b[ndim_b - 1 - i] : 1

        if dim_a == dim_b {
            result[ndim_out - 1 - i] = dim_a
        } else if dim_a == 1 {
            result[ndim_out - 1 - i] = dim_b
        } else if dim_b == 1 {
            result[ndim_out - 1 - i] = dim_a
        } else {
        // Incompatible shapes
            delete(result)
            return nil
        }
    }

    return result
}

// Perform broadcasted binary operation between two tensors
// This implements NumPy-style broadcasting semantics
broadcast_binary_op :: proc(a: Tensor, b: Tensor, output: ^Tensor, op: Broadcast_Op) {
// Check output shape is valid
    broadcast_shape := check_broadcast_shapes(a.sizes, b.sizes)
    if broadcast_shape == nil {
        panic("Incompatible shapes for broadcasting")
    }
    defer delete(broadcast_shape)

    // Verify output matches broadcast shape
    assert(len(output.sizes) == len(broadcast_shape), "Output rank mismatch")
    for i in 0..<len(broadcast_shape) {
        assert(output.sizes[i] == broadcast_shape[i], "Output shape mismatch")
    }

    // Get dimensions
    ndim_out := len(broadcast_shape)
    ndim_a := len(a.sizes)
    ndim_b := len(b.sizes)

    // Iterate through all output elements
    n_elements := len(output.data)
    indices := make([]int, ndim_out)
    defer delete(indices)

    for elem_idx in 0..<n_elements {
    // Convert linear index to multi-dimensional indices
        temp_idx := elem_idx
        for i := ndim_out - 1; i >= 0; i -= 1 {
            indices[i] = temp_idx % broadcast_shape[i]
            temp_idx /= broadcast_shape[i]
        }

        // Compute indices for input a (with broadcasting)
        idx_a := 0
        for i in 0..<ndim_out {
            dim_in_a := i - (ndim_out - ndim_a)
            if dim_in_a >= 0 {
            // This dimension exists in a
                size_a := a.sizes[dim_in_a]
                // If size is 1, broadcast (use index 0), otherwise use actual index
                actual_idx := indices[i] if size_a > 1 else 0
                idx_a += actual_idx * a.strides[dim_in_a]
            }
        }

        // Compute indices for input b (with broadcasting)
        idx_b := 0
        for i in 0..<ndim_out {
            dim_in_b := i - (ndim_out - ndim_b)
            if dim_in_b >= 0 {
            // This dimension exists in b
                size_b := b.sizes[dim_in_b]
                // If size is 1, broadcast (use index 0), otherwise use actual index
                actual_idx := indices[i] if size_b > 1 else 0
                idx_b += actual_idx * b.strides[dim_in_b]
            }
        }

        // Compute output index
        idx_out := compute_index(indices, output.strides)

        // Apply operation
        output.data[idx_out] = apply_op(a.data[idx_a], b.data[idx_b], op)
    }
}

// Broadcasting operation type
Broadcast_Op :: enum {
    Add,
    Mul,
    Sub,
    Div,
}

// Apply a binary operation to two scalars
apply_op :: proc(a, b: f32, op: Broadcast_Op) -> f32 {
    switch op {
    case .Add:
        return a + b
    case .Mul:
        return a * b
    case .Sub:
        return a - b
    case .Div:
        return a / b
    }
    return 0
}
