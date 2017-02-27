package cifar

import (
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// SampleList is an *anyff.SampleList for a list of CIFAR
// samples.
type SampleList struct {
	Samples []*Sample
	Creator anyvec.Creator

	// If true, then the samples must be from CIFAR-100 and
	// the super-classes will be used as labels.
	UseSuper bool
}

// NewSampleListAll creates a SampleList by combining all
// of the batches.
func NewSampleListAll(c anyvec.Creator, samples ...[]*Sample) *SampleList {
	var s []*Sample
	for _, x := range samples {
		s = append(s, x...)
	}
	return &SampleList{Samples: s, Creator: c}
}

// Len returns the length of the sample list.
func (s *SampleList) Len() int {
	return len(s.Samples)
}

// Swap swaps two samples.
func (s *SampleList) Swap(i, j int) {
	s.Samples[i], s.Samples[j] = s.Samples[j], s.Samples[i]
}

// Slice copies a slice of the list.
func (s *SampleList) Slice(i, j int) anysgd.SampleList {
	return &SampleList{
		Samples:  append([]*Sample{}, s.Samples[i:j]...),
		Creator:  s.Creator,
		UseSuper: s.UseSuper,
	}
}

// GetSample gets a feed-forward training sample.
func (s *SampleList) GetSample(i int) *anyff.Sample {
	sample := s.Samples[i]
	vector := make([]float64, 0, 1024*3)
	for i := 0; i < 1024; i++ {
		r, g, b := sample.Red[i], sample.Green[i], sample.Blue[i]
		for _, by := range []byte{r, g, b} {
			vector = append(vector, float64(by)/0xff)
		}
	}
	label := s.labelVector(i)
	return &anyff.Sample{
		Input:  s.Creator.MakeVectorData(s.Creator.MakeNumericList(vector)),
		Output: s.Creator.MakeVectorData(s.Creator.MakeNumericList(label)),
	}
}

func (s *SampleList) labelVector(i int) []float64 {
	sample := s.Samples[i]
	var idx, total int
	if !s.UseSuper {
		idx = sample.Class.Index
		if sample.Superclass == nil {
			total = len(cifar10Labels)
		} else {
			total = len(fineLabels)
		}
	} else {
		idx = sample.Superclass.Index
		total = len(coarseLabels)
	}
	res := make([]float64, total)
	res[idx] = 1
	return res
}
