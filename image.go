package cifar

import (
	"image"
	"image/color"
)

// Image converts a Sample to an image.
func Image(s *Sample) image.Image {
	res := image.NewRGBA(image.Rect(0, 0, 32, 32))
	var idx int
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			r, g, b := s.Red[idx], s.Green[idx], s.Blue[idx]
			res.SetRGBA(x, y, color.RGBA{
				R: r,
				G: g,
				B: b,
				A: 0xff,
			})
			idx++
		}
	}
	return res
}
