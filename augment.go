package cifar

import (
	"math/rand"

	"github.com/unixpickle/essentials"
)

// Augment applies data augmentation to a sample by
// manipulating it in subtle ways.
func Augment(sample *Sample) *Sample {
	res := *sample

	if rand.Intn(2) == 0 {
		mirror(&res.Red)
		mirror(&res.Green)
		mirror(&res.Blue)
	}

	// Vector from https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4.
	power := rand.NormFloat64()
	colorAugment(&res.Red, 0.0148366*power)
	colorAugment(&res.Green, 0.01253134*power)
	colorAugment(&res.Blue, 0.01040762*power)

	return &res
}

func mirror(channel *[1024]byte) {
	for y := 0; y < 32; y++ {
		for x := 0; x < 16; x++ {
			idx1 := y*32 + (31 - x)
			idx2 := y*32 + x
			channel[idx1], channel[idx2] = channel[idx2], channel[idx1]
		}
	}
}

func colorAugment(channel *[1024]byte, amount float64) {
	for i, x := range channel {
		num := amount + float64(x)/0xff
		val := int(num * 0x100)
		val = essentials.MinInt(0xff, essentials.MaxInt(0, val))
		channel[i] = byte(val)
	}
}
