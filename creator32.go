package cudavec

import "github.com/unixpickle/anyvec"

// A Creator32 is an anyvec.Creator for vectors using
// float32 numerics and []float32 slice types.
type Creator32 struct {
	Handle *Handle
}

// MakeNumeric creates a float32.
func (c *Creator32) MakeNumeric(x float64) anyvec.Numeric {
	return float32(x)
}

// MakeNumericList creates a []float32.
func (c *Creator32) MakeNumericList(x []float64) anyvec.NumericList {
	res := make([]float32, len(x))
	for i, k := range x {
		res[i] = float32(k)
	}
	return res
}

// MakeVector creates a zero'd out anyvec.Vector.
func (c *Creator32) MakeVector(size int) anyvec.Vector {
	panic("nyi")
}

// MakeVectorData creates an anyvec.Vector with the
// specified contents.
func (c *Creator32) MakeVectorData(dObj anyvec.NumericList) anyvec.Vector {
	panic("nyi")
}

// Concat concatenates vectors.
func (c *Creator32) Concat(v ...anyvec.Vector) anyvec.Vector {
	panic("nyi")
}

// MakeMapper creates a mapper.
func (c *Creator32) MakeMapper(inSize int, table []int) anyvec.Mapper {
	panic("nyi")
}
