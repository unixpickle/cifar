package cifar

import (
	"github.com/unixpickle/anynet"
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
func (s *SampleList) GetSample(i int) (*anyff.Sample, error) {
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
	}, nil
}

// Accuracy computes the layer's classification accuracy.
//
// The layer should have one output per class.
// The maximum output is the chosen classification.
func (s *SampleList) Accuracy(l anynet.Layer, batchSize int) anyvec.Numeric {
	fetcher := &anyff.Trainer{}
	correctSum := s.Creator.MakeVector(1)
	for i := 0; i < s.Len(); i += batchSize {
		if batchSize > s.Len()-i {
			batchSize = s.Len() - i
		}
		b, _ := fetcher.Fetch(s.Slice(i, i+batchSize))
		ins := b.(*anyff.Batch).Inputs
		desired := b.(*anyff.Batch).Outputs.Output()
		outs := l.Apply(ins, batchSize).Output()

		mapper := anyvec.MapMax(outs, outs.Len()/batchSize)
		maxes := s.Creator.MakeVector(desired.Len())
		ones := s.Creator.MakeVector(batchSize)
		ones.AddScaler(s.Creator.MakeNumeric(1))
		mapper.MapTranspose(ones, maxes)

		correctSum.AddScaler(maxes.Dot(desired))
	}
	correctSum.Scale(s.Creator.MakeNumeric(1 / float64(s.Len())))
	return anyvec.Sum(correctSum)
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
