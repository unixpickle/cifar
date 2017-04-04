package cifar

import (
	"math/rand"

	"github.com/unixpickle/essentials"
)

const augmentScaleBorder = 3

// Augment applies data augmentation to a sample by
// manipulating it in subtle ways.
func Augment(sample *Sample) *Sample {
	res := *sample

	if rand.Intn(2) == 0 {
		mirror(&res.Red)
		mirror(&res.Green)
		mirror(&res.Blue)
	}

	cropSize := (32 - augmentScaleBorder) + rand.Float64()*augmentScaleBorder
	cropX := rand.Float64() * (32 - cropSize)
	cropY := rand.Float64() * (32 - cropSize)

	colorPower := rand.NormFloat64()

	// Vector from https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4.
	colorVec := []float64{0.0148366, 0.01253134, 0.01040762}

	for i, channel := range []*[1024]byte{&res.Red, &res.Green, &res.Blue} {
		cropAugment(channel, cropX, cropY, cropSize)
		colorAugment(channel, colorPower*colorVec[i])
	}

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

func cropAugment(channel *[1024]byte, cropX, cropY, cropSize float64) {
	scale := cropSize / 32
	var res [1024]byte
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			res[x+y*32] = getInterp(channel, float64(x)*scale+cropX,
				float64(y)*scale+cropY)
		}
	}
	copy(channel[:], res[:])
}

func getInterp(channel *[1024]byte, x, y float64) byte {
	x = clipCoord(x)
	y = clipCoord(y)
	x0 := int(x)
	x1 := int(x + 1)
	y0 := int(y)
	y1 := int(y + 1)

	if x1 > 31 {
		x1 = 31
	}
	if y1 > 31 {
		y1 = 31
	}

	amountX0 := float64(x1) - x
	amountY0 := float64(y1) - y

	return toByte(getPixel(channel, x0, y0)*amountX0*amountY0 +
		getPixel(channel, x1, y0)*(1-amountX0)*amountY0 +
		getPixel(channel, x0, y1)*amountX0*(1-amountY0) +
		getPixel(channel, x1, y1)*(1-amountX0)*(1-amountY0))
}

func getPixel(channel *[1024]byte, x, y int) float64 {
	return float64(channel[x+y*32])
}

func clipCoord(x float64) float64 {
	if x < 0 {
		return 0
	} else if x > 31 {
		return 31
	} else {
		return x
	}
}

func toByte(x float64) byte {
	if x < 0 {
		return 0
	} else if x >= 0x100 {
		return 0xff
	} else {
		return byte(x)
	}
}
