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

