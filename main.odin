#+feature dynamic-literals
package main

import "core:fmt"
import "core:time"
import "core:os"

main :: proc() {
    sw: time.Stopwatch
    time.stopwatch_start(&sw)
	test_loaded_model()
    time.stopwatch_stop(&sw)
    fmt.println("Duration:", time.stopwatch_duration(sw))

	// 2D array (fixed size)
	tensor_fixed: [3][4]f32 = {
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
		{9.0, 10.0, 11.0, 12.0},
	}


	fmt.println("\n2D Fixed Array (3x4):")
	for row, i in tensor_fixed {
		fmt.println("Row", i, ":", row)
	}
    fmt.println("Element [1][2]:", tensor_fixed[1][2])

    linear := linear_init(15, 25)
    defer linear_destroy(&linear)
    /*
    model, ok := load_pytorch_model("model")
    print_weights(model)
    */
}
