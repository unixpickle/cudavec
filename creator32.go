package cudavec

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/cuda"
)

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
	return &vector32{
		creator: c,
		size:    size,
	}
}

// MakeVectorData creates an anyvec.Vector with the
// specified contents.
func (c *Creator32) MakeVectorData(list anyvec.NumericList) anyvec.Vector {
	slice := list.([]float32)
	res := c.MakeVector(len(slice))
	res.SetData(slice)
	return res
}

// Concat concatenates vectors.
func (c *Creator32) Concat(v ...anyvec.Vector) anyvec.Vector {
	totalLen := 0
	for _, x := range v {
		// Type assertion to ensure we panic during the call if
		// the type is bad.
		totalLen += x.(*vector32).Len()

		// Integer overflow.
		if totalLen < 0 {
			panic("concatenated size is too long")
		}
	}

	res := &vector32{
		creator: c,
		size:    totalLen,
	}

	c.run(func() error {
		buf, err := cuda.AllocBuffer(c.Handle.allocator, uintptr(totalLen)*4)
		if err != nil {
			return err
		}
		var off uintptr
		for _, x := range v {
			subSlice := cuda.Slice(buf, off, off+uintptr(x.Len()))
			rawX := x.(*vector32)
			if rawX.buffer != nil {
				if err := cuda.CopyBuffer(subSlice, rawX.buffer); err != nil {
					return err
				}
			} else {
				if err := cuda.ClearBuffer(subSlice); err != nil {
					return err
				}
			}
			off += uintptr(x.Len()) * 4
		}
		res.buffer = buf
		return nil
	})

	return res
}

// MakeMapper creates a mapper.
func (c *Creator32) MakeMapper(inSize int, table []int) anyvec.Mapper {
	panic("nyi")
}

func (c *Creator32) run(f func() error) <-chan error {
	return c.Handle.context.Run(func() error {
		if err := f(); err != nil {
			panic(err)
		}
		return nil
	})
}

func (c *Creator32) runSync(f func() error) {
	<-c.run(f)
}
