// Package cifar reads and processes the CIFAR-10 and
// CIFAR-100 image classification datasets.
package cifar

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/unixpickle/essentials"
)

// Class represents a category.
type Class struct {
	Name  string
	Index int
}

// Sample is a labeled 32-by-32 image.
type Sample struct {
	Class *Class

	// Non-nil for CIFAR-100
	Superclass *Class

	// Row-major images.
	Red   [1024]byte
	Green [1024]byte
	Blue  [1024]byte
}

// Load10 loads the CIFAR-10 dataset from a directory.
//
// Each batch is returned as a slice of samples.
// The training batches are first (in order), followed by
// the testing batch.
func Load10(dirPath string) ([][]*Sample, error) {
	var batchNames []string
	for i := 1; i <= 5; i++ {
		batchNames = append(batchNames, fmt.Sprintf("data_batch_%d.bin", i))
	}
	batchNames = append(batchNames, "test_batch.bin")

	var res [][]*Sample
	for _, batch := range batchNames {
		path := filepath.Join(dirPath, batch)
		samples, err := loadBatch(false, path)
		if err != nil {
			return nil, essentials.AddCtx("load "+path, err)
		}
		res = append(res, samples)
	}
	return res, nil
}

// Load100 loads the CIFAR-100 dataset from a directory.
//
// Two batches are returned (in order): training and
// testing.
func Load100(dirPath string) ([][]*Sample, error) {
	var res [][]*Sample
	for _, batch := range []string{"train.bin", "test.bin"} {
		path := filepath.Join(dirPath, batch)
		samples, err := loadBatch(true, path)
		if err != nil {
			return nil, essentials.AddCtx("load "+path, err)
		}
		res = append(res, samples)
	}
	return res, nil
}

func loadBatch(twoLabels bool, path string) ([]*Sample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	reader := bufio.NewReader(f)
	samples := []*Sample{}
	for {
		coarse, err := reader.ReadByte()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		var fine byte
		if twoLabels {
			fine, err = reader.ReadByte()
			if err != nil {
				return nil, err
			}
		}
		pixels := make([]byte, 1024*3)
		if _, err := io.ReadFull(reader, pixels); err != nil {
			return nil, err
		}

		sample := &Sample{}
		copy(sample.Red[:], pixels)
		copy(sample.Green[:], pixels[1024:])
		copy(sample.Blue[:], pixels[2048:])

		if twoLabels {
			if int(fine) >= len(fineLabels) || int(coarse) >= len(coarseLabels) {
				return nil, errors.New("label out of range")
			}
			sample.Class = &Class{
				Index: int(fine),
				Name:  fineLabels[int(fine)],
			}
			sample.Superclass = &Class{
				Index: int(coarse),
				Name:  coarseLabels[int(coarse)],
			}
		} else {
			if int(coarse) >= len(cifar10Labels) {
				return nil, errors.New("label out of range")
			}
			sample.Class = &Class{
				Index: int(coarse),
				Name:  cifar10Labels[int(coarse)],
			}
		}

		samples = append(samples, sample)
	}
	return samples, nil
}

var cifar10Labels = []string{
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck",
}

var coarseLabels = []string{
	"aquatic_mammals",
	"fish",
	"flowers",
	"food_containers",
	"fruit_and_vegetables",
	"household_electrical_devices",
	"household_furniture",
	"insects",
	"large_carnivores",
	"large_man-made_outdoor_things",
	"large_natural_outdoor_scenes",
	"large_omnivores_and_herbivores",
	"medium_mammals",
	"non-insect_invertebrates",
	"people",
	"reptiles",
	"small_mammals",
	"trees",
	"vehicles_1",
	"vehicles_2",
}

var fineLabels = []string{
	"apple",
	"aquarium_fish",
	"baby",
	"bear",
	"beaver",
	"bed",
	"bee",
	"beetle",
	"bicycle",
	"bottle",
	"bowl",
	"boy",
	"bridge",
	"bus",
	"butterfly",
	"camel",
	"can",
	"castle",
	"caterpillar",
	"cattle",
	"chair",
	"chimpanzee",
	"clock",
	"cloud",
	"cockroach",
	"couch",
	"crab",
	"crocodile",
	"cup",
	"dinosaur",
	"dolphin",
	"elephant",
	"flatfish",
	"forest",
	"fox",
	"girl",
	"hamster",
	"house",
	"kangaroo",
	"keyboard",
	"lamp",
	"lawn_mower",
	"leopard",
	"lion",
	"lizard",
	"lobster",
	"man",
	"maple_tree",
	"motorcycle",
	"mountain",
	"mouse",
	"mushroom",
	"oak_tree",
	"orange",
	"orchid",
	"otter",
	"palm_tree",
	"pear",
	"pickup_truck",
	"pine_tree",
	"plain",
	"plate",
	"poppy",
	"porcupine",
	"possum",
	"rabbit",
	"raccoon",
	"ray",
	"road",
	"rocket",
	"rose",
	"sea",
	"seal",
	"shark",
	"shrew",
	"skunk",
	"skyscraper",
	"snail",
	"snake",
	"spider",
	"squirrel",
	"streetcar",
	"sunflower",
	"sweet_pepper",
	"table",
	"tank",
	"telephone",
	"television",
	"tiger",
	"tractor",
	"train",
	"trout",
	"tulip",
	"turtle",
	"wardrobe",
	"whale",
	"willow_tree",
	"wolf",
	"woman",
	"worm",
}
