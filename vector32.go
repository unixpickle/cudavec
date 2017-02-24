package cudavec

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/cuda"
)

type vector32 struct {
	creator *Creator32
	size    int

	// May be nil for lazy evaluations.
	buffer cuda.Buffer
}

func (v *vector32) Creator() anyvec.Creator {
	return v.creator
}

func (v *vector32) Len() int {
	return v.size
}

func (v *vector32) Data() anyvec.NumericList {
	res := make([]float32, v.Len())
	v.runSync(func() error {
		if v.buffer != nil {
			return cuda.ReadBuffer(res, v.buffer)
		}
		return nil
	})
	return res
}

func (v *vector32) SetData(d anyvec.NumericList) {
	slice := d.([]float32)
	if len(slice) > v.Len() {
		panic("index out of bounds")
	}
	v.runSync(func() error {
		if err := v.lazyInit(len(slice) < v.Len()); err != nil {
			return err
		}
		return cuda.WriteBuffer(v.buffer, slice)
	})
}

func (v *vector32) Scale(s anyvec.Numeric) {
	v.run(func() error {
		if v.buffer == nil {
			return nil
		}
		return v.creator.Handle.blas.Sscal(v.Len(), s.(float32), v.buffer, 1)
	})
}

func (v *vector32) run(f func() error) <-chan error {
	return v.creator.Handle.context.Run(func() error {
		if err := f(); err != nil {
			panic(err)
		}
		return nil
	})
}

func (v *vector32) runSync(f func() error) {
	<-v.run(f)
}

func (v *vector32) lazyInit(clear bool) error {
	if v.buffer != nil {
		return nil
	}
	var err error
	v.buffer, err = cuda.AllocBuffer(v.creator.Handle.allocator, uintptr(v.Len())*4)
	if err != nil {
		return err
	}
	if clear {
		return cuda.ClearBuffer(v.buffer)
	}
	return nil
}
