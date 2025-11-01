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
}
