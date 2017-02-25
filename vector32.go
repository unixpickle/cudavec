package cudavec

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
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
		panic("index out of range")
	}
	v.runSync(func() error {
		if err := v.lazyInit(len(slice) < v.Len()); err != nil {
			return err
		}
		return cuda.WriteBuffer(v.buffer, slice)
	})
}

func (v *vector32) Set(v1 anyvec.Vector) {
	v.assertCompat(v1, false)
	v.run(func() error {
		buf1 := v1.(*vector32).buffer
		if buf1 == nil {
			if v.buffer != nil {
				return cuda.ClearBuffer(v.buffer)
			}
			return nil
		}
		if err := v.lazyInit(false); err != nil {
			return err
		}
		return cuda.CopyBuffer(v.buffer, buf1)
	})
}

func (v *vector32) Copy() anyvec.Vector {
	v1 := v.Creator().MakeVector(v.Len())
	v1.Set(v)
	return v1
}

func (v *vector32) Slice(start, end int) anyvec.Vector {
	if start < 0 || start > end || end > v.Len() {
		panic("index out of range")
	}
	res := &vector32{creator: v.creator, size: end - start}
	v.run(func() (err error) {
		if v.buffer == nil {
			return nil
		}
		subSlice := cuda.Slice(v.buffer, uintptr(start)*4, uintptr(end)*4)
		res.buffer, err = cuda.AllocBuffer(v.creator.Handle.allocator,
			uintptr(res.size)*4)
		if err != nil {
			return err
		}
		return cuda.CopyBuffer(res.buffer, subSlice)
	})
	return res
}

func (v *vector32) SetSlice(start int, v1 anyvec.Vector) {
	src := v1.(*vector32)
	if src.Len() > v.Len()-start {
		panic("index out of range")
	} else if start <= -src.Len() {
		return
	}

	v.run(func() error {
		if src.buffer == nil && v.buffer == nil {
			return nil
		}
		dstStart := start
		srcStart := 0
		if start < 0 {
			dstStart = 0
			srcStart = -start
		}
		copyCount := src.Len() - srcStart
		if v.Len()-dstStart < copyCount {
			copyCount = v.Len() - dstStart
		}
		if err := v.lazyInit(true); err != nil {
			return err
		}
		dst := cuda.Slice(v.buffer, uintptr(dstStart)*4,
			uintptr(dstStart+copyCount)*4)
		if src.buffer == nil {
			return cuda.ClearBuffer(dst)
		}
		srcSlice := cuda.Slice(src.buffer, uintptr(srcStart)*4,
			uintptr(srcStart+copyCount)*4)
		return cuda.CopyBuffer(dst, srcSlice)
	})
}

func (v *vector32) Scale(s anyvec.Numeric) {
	scaler := s.(float32)
	v.run(func() error {
		if v.buffer == nil {
			return nil
		}
		return v.creator.Handle.blas.Sscal(v.Len(), scaler, v.buffer, 1)
	})
}

func (v *vector32) AddScaler(s anyvec.Numeric) {
	panic("nyi")
}

func (v *vector32) Dot(v1 anyvec.Vector) anyvec.Numeric {
	v.assertCompat(v1, true)
	var res float32
	v.runSync(func() error {
		v32 := v1.(*vector32)
		if err := lazyInitAll(true, v, v32); err != nil {
			return err
		}
		return v.creator.Handle.blas.Sdot(v.Len(), v.buffer, 1, v32.buffer, 1, &res)
	})
	return res
}

func (v *vector32) Add(v1 anyvec.Vector) {
	v.axpy(1, v1)
}

func (v *vector32) Sub(v1 anyvec.Vector) {
	v.axpy(-1, v1)
}

func (v *vector32) Mul(v1 anyvec.Vector) {
	v.assertCompat(v1, false)
	panic("nyi")
}

func (v *vector32) Div(v1 anyvec.Vector) {
	v.assertCompat(v1, false)
	v132 := v1.(*vector32)
	v.run(func() error {
		if err := lazyInitAll(true, v, v132); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch("kernel", grid, 1, 1,
			block, 1, 1, 0, v.buffer, v132.buffer, v.Len())
	})
}

func (v *vector32) Gemm(transA, transB bool, m, n, k int,
	alpha anyvec.Numeric, a anyvec.Vector, lda int,
	b anyvec.Vector, ldb int, beta anyvec.Numeric, ldc int) {
	alphaFloat := alpha.(float32)
	betaFloat := beta.(float32)
	a32 := a.(*vector32)
	b32 := b.(*vector32)
	if a32 == b32 || a32 == v || b32 == v {
		panic("vectors cannot be equal")
	}
	v.run(func() error {
		if err := lazyInitAll(true, v, a32, b32); err != nil {
			return err
		}
		ta := cublas.NoTrans
		tb := cublas.NoTrans
		if transA {
			ta = cublas.Trans
		}
		if transB {
			tb = cublas.Trans
		}
		return v.creator.Handle.blas.Sgemm(ta, tb, m, n, k,
			alphaFloat, a32.buffer, lda, b32.buffer, ldb,
			betaFloat, v.buffer, ldc)
	})
}

func (v *vector32) axpy(scaler float32, v1 anyvec.Vector) {
	v.assertCompat(v1, false)
	v32 := v1.(*vector32)
	v.run(func() error {
		if v32.buffer == nil {
			return nil
		} else if v.buffer == nil {
			if err := v.lazyInit(false); err != nil {
				return err
			}
			if err := cuda.CopyBuffer(v.buffer, v32.buffer); err != nil {
				return err
			}
			if scaler == 1 {
				return nil
			}
			return v.creator.Handle.blas.Sscal(v.Len(), scaler, v.buffer, 1)
		}
		return v.creator.Handle.blas.Saxpy(v.Len(), scaler, v32.buffer, 1,
			v.buffer, 1)
	})
}

func (v *vector32) run(f func() error) <-chan error {
	return v.creator.run(f)
}

func (v *vector32) runSync(f func() error) {
	v.creator.runSync(f)
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

func (v *vector32) assertCompat(v1 anyvec.Vector, readOnly bool) {
	v132 := v1.(*vector32)
	if readOnly || v == v132 {
		panic("vectors cannot be equal")
	} else if v.Len() != v132.Len() {
		panic("length mismatch")
	}
}

func (v *vector32) kernelSizes() (grid, block uint) {
	block = 128
	if uint(v.Len()) < block {
		block = uint(v.Len())
		grid = 1
	} else {
		grid = uint(v.Len()) / block
		if uint(v.Len())%block != 0 {
			grid++
		}
	}
	return
}

func lazyInitAll(clear bool, vs ...*vector32) error {
	for _, x := range vs {
		if err := x.lazyInit(clear); err != nil {
			return err
		}
	}
	return nil
}
