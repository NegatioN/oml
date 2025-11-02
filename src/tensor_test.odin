package main

import "core:testing"
import "core:fmt"

// Helper to create a tensor with given data, sizes, and automatic contiguous strides
make_tensor :: proc(data: []f32, sizes: []int) -> Tensor {
    strides := make([]int, len(sizes))
    stride := 1
    for i := len(sizes) - 1; i >= 0; i -= 1 {
        strides[i] = stride
        stride *= sizes[i]
    }
    
    return Tensor{
        data = data,
        sizes = sizes,
        strides = strides,
    }
}

// Helper to clean up a tensor created with make_tensor
cleanup_tensor :: proc(t: Tensor) {
    delete(t.strides)
}

// Test 1: Scalar + Scalar (both shape [1])
@(test)
test_broadcast_scalar_scalar :: proc(t: ^testing.T) {
    a_data := []f32{5.0}
    a := make_tensor(a_data, []int{1})
    defer cleanup_tensor(a)
    
    b_data := []f32{3.0}
    b := make_tensor(b_data, []int{1})
    defer cleanup_tensor(b)
    
    out_data := make([]f32, 1)
    defer delete(out_data)
    out := make_tensor(out_data, []int{1})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Add)
    
    testing.expect_value(t, out.data[0], 8.0)
}

// Test 2: 1D + Scalar broadcasting
@(test)
test_broadcast_1d_scalar :: proc(t: ^testing.T) {
    a_data := []f32{1.0, 2.0, 3.0, 4.0}
    a := make_tensor(a_data, []int{4})
    defer cleanup_tensor(a)
    
    b_data := []f32{10.0}
    b := make_tensor(b_data, []int{1})
    defer cleanup_tensor(b)
    
    out_data := make([]f32, 4)
    defer delete(out_data)
    out := make_tensor(out_data, []int{4})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Add)
    
    testing.expect_value(t, out.data[0], 11.0)
    testing.expect_value(t, out.data[1], 12.0)
    testing.expect_value(t, out.data[2], 13.0)
    testing.expect_value(t, out.data[3], 14.0)
}

// Test 3: Scalar + 1D broadcasting (reversed)
@(test)
test_broadcast_scalar_1d :: proc(t: ^testing.T) {
    a_data := []f32{100.0}
    a := make_tensor(a_data, []int{1})
    defer cleanup_tensor(a)
    
    b_data := []f32{1.0, 2.0, 3.0}
    b := make_tensor(b_data, []int{3})
    defer cleanup_tensor(b)
    
    out_data := make([]f32, 3)
    defer delete(out_data)
    out := make_tensor(out_data, []int{3})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Mul)
    
    testing.expect_value(t, out.data[0], 100.0)
    testing.expect_value(t, out.data[1], 200.0)
    testing.expect_value(t, out.data[2], 300.0)
}

// Test 4: 1D + 1D same size (element-wise)
@(test)
test_broadcast_1d_1d_same :: proc(t: ^testing.T) {
    a_data := []f32{1.0, 2.0, 3.0}
    a := make_tensor(a_data, []int{3})
    defer cleanup_tensor(a)
    
    b_data := []f32{10.0, 20.0, 30.0}
    b := make_tensor(b_data, []int{3})
    defer cleanup_tensor(b)
    
    out_data := make([]f32, 3)
    defer delete(out_data)
    out := make_tensor(out_data, []int{3})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Add)
    
    testing.expect_value(t, out.data[0], 11.0)
    testing.expect_value(t, out.data[1], 22.0)
    testing.expect_value(t, out.data[2], 33.0)
}

// Test 5: 2D + 1D broadcasting (broadcasting along last dimension)
@(test)
test_broadcast_2d_1d :: proc(t: ^testing.T) {
    // Shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
    a_data := []f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    a := make_tensor(a_data, []int{2, 3})
    defer cleanup_tensor(a)
    
    // Shape [3]: [10, 20, 30]
    b_data := []f32{10.0, 20.0, 30.0}
    b := make_tensor(b_data, []int{3})
    defer cleanup_tensor(b)
    
    // Output shape [2, 3]
    out_data := make([]f32, 6)
    defer delete(out_data)
    out := make_tensor(out_data, []int{2, 3})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Add)
    
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    testing.expect_value(t, out.data[0], 11.0)
    testing.expect_value(t, out.data[1], 22.0)
    testing.expect_value(t, out.data[2], 33.0)
    // Row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
    testing.expect_value(t, out.data[3], 14.0)
    testing.expect_value(t, out.data[4], 25.0)
    testing.expect_value(t, out.data[5], 36.0)
}

// Test 6: 2D + 2D with broadcasting on first dimension
@(test)
test_broadcast_2d_2d_first_dim :: proc(t: ^testing.T) {
    // Shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
    a_data := []f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    a := make_tensor(a_data, []int{2, 3})
    defer cleanup_tensor(a)
    
    // Shape [1, 3]: [[10, 20, 30]] (broadcasts along first dimension)
    b_data := []f32{10.0, 20.0, 30.0}
    b := make_tensor(b_data, []int{1, 3})
    defer cleanup_tensor(b)
    
    // Output shape [2, 3]
    out_data := make([]f32, 6)
    defer delete(out_data)
    out := make_tensor(out_data, []int{2, 3})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Mul)
    
    // Row 0: [1*10, 2*20, 3*30] = [10, 40, 90]
    testing.expect_value(t, out.data[0], 10.0)
    testing.expect_value(t, out.data[1], 40.0)
    testing.expect_value(t, out.data[2], 90.0)
    // Row 1: [4*10, 5*20, 6*30] = [40, 100, 180]
    testing.expect_value(t, out.data[3], 40.0)
    testing.expect_value(t, out.data[4], 100.0)
    testing.expect_value(t, out.data[5], 180.0)
}

// Test 7: 2D + 2D with broadcasting on second dimension
@(test)
test_broadcast_2d_2d_second_dim :: proc(t: ^testing.T) {
    // Shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
    a_data := []f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    a := make_tensor(a_data, []int{2, 3})
    defer cleanup_tensor(a)
    
    // Shape [2, 1]: [[10], [20]] (broadcasts along second dimension)
    b_data := []f32{10.0, 20.0}
    b := make_tensor(b_data, []int{2, 1})
    defer cleanup_tensor(b)
    
    // Output shape [2, 3]
    out_data := make([]f32, 6)
    defer delete(out_data)
    out := make_tensor(out_data, []int{2, 3})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Add)
    
    // Row 0: [1+10, 2+10, 3+10] = [11, 12, 13]
    testing.expect_value(t, out.data[0], 11.0)
    testing.expect_value(t, out.data[1], 12.0)
    testing.expect_value(t, out.data[2], 13.0)
    // Row 1: [4+20, 5+20, 6+20] = [24, 25, 26]
    testing.expect_value(t, out.data[3], 24.0)
    testing.expect_value(t, out.data[4], 25.0)
    testing.expect_value(t, out.data[5], 26.0)
}

// Test 8: 3D + 2D broadcasting
@(test)
test_broadcast_3d_2d :: proc(t: ^testing.T) {
    // Shape [2, 2, 2]: two 2x2 matrices
    // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    a_data := []f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
    a := make_tensor(a_data, []int{2, 2, 2})
    defer cleanup_tensor(a)
    
    // Shape [2, 2]: [[10, 20], [30, 40]]
    b_data := []f32{10.0, 20.0, 30.0, 40.0}
    b := make_tensor(b_data, []int{2, 2})
    defer cleanup_tensor(b)
    
    // Output shape [2, 2, 2]
    out_data := make([]f32, 8)
    defer delete(out_data)
    out := make_tensor(out_data, []int{2, 2, 2})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Add)
    
    // First matrix: [[1+10, 2+20], [3+30, 4+40]] = [[11, 22], [33, 44]]
    testing.expect_value(t, out.data[0], 11.0)
    testing.expect_value(t, out.data[1], 22.0)
    testing.expect_value(t, out.data[2], 33.0)
    testing.expect_value(t, out.data[3], 44.0)
    // Second matrix: [[5+10, 6+20], [7+30, 8+40]] = [[15, 26], [37, 48]]
    testing.expect_value(t, out.data[4], 15.0)
    testing.expect_value(t, out.data[5], 26.0)
    testing.expect_value(t, out.data[6], 37.0)
    testing.expect_value(t, out.data[7], 48.0)
}

// Test 9: 3D + 1D broadcasting
@(test)
test_broadcast_3d_1d :: proc(t: ^testing.T) {
    // Shape [2, 2, 2]
    a_data := []f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
    a := make_tensor(a_data, []int{2, 2, 2})
    defer cleanup_tensor(a)
    
    // Shape [2]: [100, 200] (broadcasts along last dimension)
    b_data := []f32{100.0, 200.0}
    b := make_tensor(b_data, []int{2})
    defer cleanup_tensor(b)
    
    // Output shape [2, 2, 2]
    out_data := make([]f32, 8)
    defer delete(out_data)
    out := make_tensor(out_data, []int{2, 2, 2})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Mul)
    
    // Pattern: multiply by [100, 200] repeatedly
    testing.expect_value(t, out.data[0], 100.0)  // 1*100
    testing.expect_value(t, out.data[1], 400.0)  // 2*200
    testing.expect_value(t, out.data[2], 300.0)  // 3*100
    testing.expect_value(t, out.data[3], 800.0)  // 4*200
    testing.expect_value(t, out.data[4], 500.0)  // 5*100
    testing.expect_value(t, out.data[5], 1200.0) // 6*200
    testing.expect_value(t, out.data[6], 700.0)  // 7*100
    testing.expect_value(t, out.data[7], 1600.0) // 8*200
}

// Test 10: Subtraction operation
@(test)
test_broadcast_sub :: proc(t: ^testing.T) {
    a_data := []f32{10.0, 20.0, 30.0}
    a := make_tensor(a_data, []int{3})
    defer cleanup_tensor(a)
    
    b_data := []f32{3.0}
    b := make_tensor(b_data, []int{1})
    defer cleanup_tensor(b)
    
    out_data := make([]f32, 3)
    defer delete(out_data)
    out := make_tensor(out_data, []int{3})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Sub)
    
    testing.expect_value(t, out.data[0], 7.0)   // 10 - 3
    testing.expect_value(t, out.data[1], 17.0)  // 20 - 3
    testing.expect_value(t, out.data[2], 27.0)  // 30 - 3
}

// Test 11: Division operation
@(test)
test_broadcast_div :: proc(t: ^testing.T) {
    a_data := []f32{10.0, 20.0, 30.0, 40.0}
    a := make_tensor(a_data, []int{4})
    defer cleanup_tensor(a)
    
    b_data := []f32{2.0}
    b := make_tensor(b_data, []int{1})
    defer cleanup_tensor(b)
    
    out_data := make([]f32, 4)
    defer delete(out_data)
    out := make_tensor(out_data, []int{4})
    defer cleanup_tensor(out)
    
    broadcast_binary_op(a, b, &out, .Div)
    
    testing.expect_value(t, out.data[0], 5.0)   // 10 / 2
    testing.expect_value(t, out.data[1], 10.0)  // 20 / 2
    testing.expect_value(t, out.data[2], 15.0)  // 30 / 2
    testing.expect_value(t, out.data[3], 20.0)  // 40 / 2
}

// Test 13: Test check_broadcast_shapes helper
@(test)
test_check_broadcast_shapes :: proc(t: ^testing.T) {
    // Compatible: [3] and [1] -> [3]
    {
        shape_a := []int{3}
        shape_b := []int{1}
        result := check_broadcast_shapes(shape_a, shape_b)
        defer delete(result)
        
        testing.expect(t, result != nil, "Shapes should be compatible")
        testing.expect_value(t, len(result), 1)
        testing.expect_value(t, result[0], 3)
    }
    
    // Compatible: [2, 3] and [3] -> [2, 3]
    {
        shape_a := []int{2, 3}
        shape_b := []int{3}
        result := check_broadcast_shapes(shape_a, shape_b)
        defer delete(result)
        
        testing.expect(t, result != nil, "Shapes should be compatible")
        testing.expect_value(t, len(result), 2)
        testing.expect_value(t, result[0], 2)
        testing.expect_value(t, result[1], 3)
    }
    
    // Compatible: [2, 1] and [1, 3] -> [2, 3]
    {
        shape_a := []int{2, 1}
        shape_b := []int{1, 3}
        result := check_broadcast_shapes(shape_a, shape_b)
        defer delete(result)
        
        testing.expect(t, result != nil, "Shapes should be compatible")
        testing.expect_value(t, len(result), 2)
        testing.expect_value(t, result[0], 2)
        testing.expect_value(t, result[1], 3)
    }
    
    // Incompatible: [3] and [2] -> nil
    {
        shape_a := []int{3}
        shape_b := []int{2}
        result := check_broadcast_shapes(shape_a, shape_b)
        
        testing.expect(t, result == nil, "Shapes should be incompatible")
    }
    
    // Incompatible: [2, 3] and [2, 2] -> nil
    {
        shape_a := []int{2, 3}
        shape_b := []int{2, 2}
        result := check_broadcast_shapes(shape_a, shape_b)
        
        testing.expect(t, result == nil, "Shapes should be incompatible")
    }
}

// ========== Mean and Standard Deviation Tests ==========

// Test: Mean of simple 1D tensor
@(test)
test_mean_simple :: proc(t: ^testing.T) {
    data := []f32{1.0, 2.0, 3.0, 4.0, 5.0}
    tensor := make_tensor(data, []int{5})
    defer cleanup_tensor(tensor)
    
    result := mean(tensor)
    expected := f32(3.0) // (1+2+3+4+5)/5 = 15/5 = 3
    
    testing.expect_value(t, result, expected)
}

// Test: Mean of all zeros
@(test)
test_mean_zeros :: proc(t: ^testing.T) {
    data := []f32{0.0, 0.0, 0.0, 0.0}
    tensor := make_tensor(data, []int{4})
    defer cleanup_tensor(tensor)
    
    result := mean(tensor)
    
    testing.expect_value(t, result, 0.0)
}

// Test: Mean of negative numbers
@(test)
test_mean_negative :: proc(t: ^testing.T) {
    data := []f32{-5.0, -3.0, -1.0, 1.0, 3.0, 5.0}
    tensor := make_tensor(data, []int{6})
    defer cleanup_tensor(tensor)
    
    result := mean(tensor)
    expected := f32(0.0) // Sum = 0, Mean = 0
    
    testing.expect_value(t, result, expected)
}

// Test: Mean with fractional values
@(test)
test_mean_fractional :: proc(t: ^testing.T) {
    data := []f32{1.5, 2.5, 3.5}
    tensor := make_tensor(data, []int{3})
    defer cleanup_tensor(tensor)
    
    result := mean(tensor)
    expected := f32(2.5) // (1.5+2.5+3.5)/3 = 7.5/3 = 2.5
    
    testing.expect_value(t, result, expected)
}

// Test: Mean using f32_mean directly
@(test)
test_f32_mean :: proc(t: ^testing.T) {
    data := []f32{10.0, 20.0, 30.0, 40.0}
    
    result := f32_mean(data)
    expected := f32(25.0) // (10+20+30+40)/4 = 100/4 = 25
    
    testing.expect_value(t, result, expected)
}

// Test: Standard deviation of constant values (should be 0)
@(test)
test_stddev_constant :: proc(t: ^testing.T) {
    data := []f32{5.0, 5.0, 5.0, 5.0, 5.0}
    tensor := make_tensor(data, []int{5})
    defer cleanup_tensor(tensor)
    
    result := stddev(tensor)
    
    testing.expect_value(t, result, 0.0)
}

// Test: Standard deviation of simple dataset
@(test)
test_stddev_simple :: proc(t: ^testing.T) {
    // Using values: [1, 2, 3, 4, 5]
    // Mean = 3
    // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
    //          = (4 + 1 + 0 + 1 + 4) / 5 = 10/5 = 2
    // StdDev = sqrt(2) ≈ 1.414
    data := []f32{1.0, 2.0, 3.0, 4.0, 5.0}
    tensor := make_tensor(data, []int{5})
    defer cleanup_tensor(tensor)
    
    result := stddev(tensor)
    expected := f32(1.4142135) // sqrt(2)
    
    // Use a small epsilon for floating point comparison
    diff := abs(result - expected)
    testing.expect(t, diff < 0.0001, "Standard deviation should be approximately sqrt(2)")
}

// Test: Standard deviation of zeros
@(test)
test_stddev_zeros :: proc(t: ^testing.T) {
    data := []f32{0.0, 0.0, 0.0}
    tensor := make_tensor(data, []int{3})
    defer cleanup_tensor(tensor)
    
    result := stddev(tensor)
    
    testing.expect_value(t, result, 0.0)
}

// Test: Standard deviation with negative values
@(test)
test_stddev_negative :: proc(t: ^testing.T) {
    // Using values: [-2, -1, 0, 1, 2]
    // Mean = 0
    // Variance = (4 + 1 + 0 + 1 + 4) / 5 = 10/5 = 2
    // StdDev = sqrt(2) ≈ 1.414
    data := []f32{-2.0, -1.0, 0.0, 1.0, 2.0}
    tensor := make_tensor(data, []int{5})
    defer cleanup_tensor(tensor)
    
    result := stddev(tensor)
    expected := f32(1.4142135)
    
    diff := abs(result - expected)
    testing.expect(t, diff < 0.0001, "Standard deviation should be approximately sqrt(2)")
}

// Test: Standard deviation with large variance
@(test)
test_stddev_large_variance :: proc(t: ^testing.T) {
    // Values with large spread
    data := []f32{0.0, 100.0}
    tensor := make_tensor(data, []int{2})
    defer cleanup_tensor(tensor)
    
    result := stddev(tensor)
    // Mean = 50
    // Variance = ((0-50)^2 + (100-50)^2) / 2 = (2500 + 2500) / 2 = 2500
    // StdDev = sqrt(2500) = 50
    expected := f32(50.0)
    
    testing.expect_value(t, result, expected)
}

// Test: Mean and stddev on same dataset
@(test)
test_mean_and_stddev_combined :: proc(t: ^testing.T) {
    data := []f32{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}
    tensor := make_tensor(data, []int{8})
    defer cleanup_tensor(tensor)
    
    m := mean(tensor)
    s := stddev(tensor)
    
    // Mean = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5
    testing.expect_value(t, m, 5.0)
    
    // Variance = ((2-5)^2 + 3*(4-5)^2 + 2*(5-5)^2 + (7-5)^2 + (9-5)^2) / 8
    //          = (9 + 3 + 0 + 4 + 16) / 8 = 32/8 = 4
    // StdDev = sqrt(4) = 2
    testing.expect_value(t, s, 2.0)
}

// Test: Mean with 2D tensor (should flatten and compute mean)
@(test)
test_mean_2d_tensor :: proc(t: ^testing.T) {
    // 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
    data := []f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    tensor := make_tensor(data, []int{2, 3})
    defer cleanup_tensor(tensor)
    
    result := mean(tensor)
    expected := f32(3.5) // (1+2+3+4+5+6)/6 = 21/6 = 3.5
    
    testing.expect_value(t, result, expected)
}

// Test: Stddev with 2D tensor
@(test)
test_stddev_2d_tensor :: proc(t: ^testing.T) {
    // 2x2 tensor: [[1, 3], [3, 5]]
    // Mean = (1+3+3+5)/4 = 12/4 = 3
    // Variance = ((1-3)^2 + (3-3)^2 + (3-3)^2 + (5-3)^2) / 4
    //          = (4 + 0 + 0 + 4) / 4 = 8/4 = 2
    // StdDev = sqrt(2) ≈ 1.414
    data := []f32{1.0, 3.0, 3.0, 5.0}
    tensor := make_tensor(data, []int{2, 2})
    defer cleanup_tensor(tensor)
    
    result := stddev(tensor)
    expected := f32(1.4142135)
    
    diff := abs(result - expected)
    testing.expect(t, diff < 0.0001, "Standard deviation should be approximately sqrt(2)")
}

// Test 13: Test with actual model operations (add_forward and sub_forward)
@(test)
test_add_tensors:: proc(t: ^testing.T) {
    a_data := []f32{1.0, 2.0, 3.0}
    a := make_tensor(a_data, []int{3})
    defer cleanup_tensor(a)
    
    b_data := []f32{10.0, 20.0, 30.0}
    b := make_tensor(b_data, []int{3})
    defer cleanup_tensor(b)
    
    out_data := make([]f32, 3)
    defer delete(out_data)
    out := make_tensor(out_data, []int{3})
    defer cleanup_tensor(out)
    
    add_tensors(a, b, &out)
    
    testing.expect_value(t, out.data[0], 11.0)
    testing.expect_value(t, out.data[1], 22.0)
    testing.expect_value(t, out.data[2], 33.0)
}

@(test)
test_sub_tensors :: proc(t: ^testing.T) {
    a_data := []f32{10.0, 20.0, 30.0}
    a := make_tensor(a_data, []int{3})
    defer cleanup_tensor(a)
    
    b_data := []f32{1.0, 2.0, 3.0}
    b := make_tensor(b_data, []int{3})
    defer cleanup_tensor(b)
    
    out_data := make([]f32, 3)
    defer delete(out_data)
    out := make_tensor(out_data, []int{3})
    defer cleanup_tensor(out)
    
    sub_tensors(a, b, &out)
    
    testing.expect_value(t, out.data[0], 9.0)   // 10 - 1
    testing.expect_value(t, out.data[1], 18.0)  // 20 - 2
    testing.expect_value(t, out.data[2], 27.0)  // 30 - 3
}
