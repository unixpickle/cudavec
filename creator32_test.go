package cudavec

import (
	"testing"

	"github.com/unixpickle/anyvec/anyvectest"
)

func TestCreator32(t *testing.T) {
	handle := setupTest(t)
	c := &Creator32{Handle: handle}
	tester := &anyvectest.Tester{Creator: c}
	tester.TestAll(t)
}

func BenchmarkCreator32(b *testing.B) {
	handle := setupTest(b)
	c := &Creator32{Handle: handle}
	bencher := &anyvectest.Bencher{Creator: c}
	bencher.BenchmarkAll(b)
}
