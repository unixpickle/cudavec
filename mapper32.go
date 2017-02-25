package cudavec

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/cuda"
)

type mapper32 struct {
	creator *Creator32
	table   cuda.Buffer
	inSize  int
	outSize int
}

func newMapper32(c *Creator32, inSize int, table []int) *mapper32 {
	if int(int32(inSize)) != inSize || int(int32(len(table))) != len(table) {
		panic("mapper size is too big")
	}
	ints32 := make([]int32, len(table))
	for i, x := range table {
		if x >= inSize || x < 0 {
			panic("index out of range")
		}
		ints32[i] = int32(x)
	}
	res := &mapper32{creator: c, inSize: inSize, outSize: len(table)}
	c.run(func() error {
		buf, err := cuda.AllocBuffer(c.Handle.allocator, uintptr(inSize)*4)
		if err != nil {
			return err
		}
		res.table = buf
		return cuda.WriteBuffer(buf, ints32)
	})
	return res
}

func (m *mapper32) Creator() anyvec.Creator {
	return m.creator
}

func (m *mapper32) InSize() int {
	return m.inSize
}

func (m *mapper32) OutSize() int {
	return m.outSize
}

func (m *mapper32) Map(in, out anyvec.Vector) {
	if in.Len() != m.inSize {
		panic("bad input size")
	} else if out.Len() != m.outSize {
		panic("bad out size")
	} else if in == out {
		panic("inputs overlap")
	}
	in32 := in.(*vector32)
	out32 := out.(*vector32)
	m.creator.run(func() error {
		if in32.buffer == nil {
			if out32.buffer != nil {
				return cuda.ClearBuffer(out32.buffer)
			}
			return nil
		}
		if err := out32.lazyInit(false); err != nil {
			return err
		}
		grid, block := out32.kernelSizes()
		return m.creator.Handle.kernels32.Launch("mapForward", grid, 1, 1, block, 1, 1,
			0, out32.buffer, in32.buffer, m.table, m.outSize)
	})
}

func (m *mapper32) MapTranspose(in, out anyvec.Vector) {
	if in.Len() != m.outSize {
		panic("bad input size")
	} else if out.Len() != m.inSize {
		panic("bad out size")
	} else if in == out {
		panic("inputs overlap")
	}
	in32 := in.(*vector32)
	out32 := out.(*vector32)
	m.creator.run(func() error {
		if err := lazyInitAll(true, in32, out32); err != nil {
			return err
		}
		grid, block := in32.kernelSizes()
		return m.creator.Handle.kernels32.Launch("mapBackward", grid, 1, 1, block, 1, 1,
			0, out32.buffer, in32.buffer, m.table, m.outSize)
	})
}
