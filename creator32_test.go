package cudavec

import (
	"testing"

	"github.com/unixpickle/anyvec/anyvectest"
)

func TestCreator32(t *testing.T) {
	handle, err := NewHandleDefault()
	if err != nil {
		t.Fatal(err)
	}
	c := &Creator32{Handle: handle}
	tester := &anyvectest.Tester{Creator: c}
	tester.TestAll(t)
}
