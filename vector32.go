package cudavec

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

type vector32 struct {
	creator *Creator32
	size    int

	// Used to detect overlap.
	bufferID *int
	start    int

	// May be nil for lazy evaluations.
	buffer cuda.Buffer
}

func (v *vector32) Creator() anyvec.Creator {
	return v.creator
}

func (v *vector32) Len() int {
	return v.size
}

func (v *vector32) Overlaps(v1 anyvec.Vector) bool {
	v1Vec := v1.(*vector32)
	return v1Vec.bufferID == v.bufferID &&
		v.start < v1Vec.start+v1Vec.Len() &&
		v1Vec.start < v.start+v.Len()
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

func (v *vector32) Set(other anyvec.Vector) {
	v1 := other.(*vector32)
	v.assertCompat(v1, false)
	v.run(func() error {
		buf1 := v1.buffer
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
	res := &vector32{
		creator:  v.creator,
		size:     end - start,
		bufferID: v.bufferID,
		start:    v.start + start,
	}
	v.run(func() (err error) {
		if err := v.lazyInit(true); err != nil {
			return err
		}
		res.buffer = cuda.Slice(v.buffer, uintptr(start)*4, uintptr(end)*4)
		return nil
	})
	return res
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
	scaler := s.(float32)
	v.run(func() error {
		if err := v.lazyInit(true); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch("addScaler", grid, 1, 1,
			block, 1, 1, 0, nil, scaler, v.buffer, v.Len())
	})
}

func (v *vector32) Dot(other anyvec.Vector) anyvec.Numeric {
	v1 := other.(*vector32)
	v.assertCompat(v1, true)
	var res float32
	v.runSync(func() error {
		if err := lazyInitAll(true, v, v1); err != nil {
			return err
		}
		return v.creator.Handle.blas.Sdot(v.Len(), v.buffer, 1, v1.buffer, 1, &res)
	})
	return res
}

func (v *vector32) Add(other anyvec.Vector) {
	v.axpy(1, other.(*vector32))
}

func (v *vector32) Sub(other anyvec.Vector) {
	v.axpy(-1, other.(*vector32))
}

func (v *vector32) Mul(other anyvec.Vector) {
	v1 := other.(*vector32)
	v.assertCompat(v1, false)
	if v.Len() == 0 {
		return
	}
	v.run(func() error {
		if err := lazyInitAll(true, v, v1); err != nil {
			return err
		}
		return v.creator.Handle.blas.Sdgmm(cublas.Left, v.Len(), 1,
			v.buffer, v.Len(), v1.buffer, 1, v.buffer, v.Len())
	})
}

func (v *vector32) Div(other anyvec.Vector) {
	v1 := other.(*vector32)
	v.assertCompat(v1, false)
	v.run(func() error {
		if err := lazyInitAll(true, v, v1); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch("divElements", grid, 1, 1,
			block, 1, 1, 0, nil, v.buffer, v1.buffer, v.Len())
	})
}

func (v *vector32) Gemm(transA, transB bool, m, n, k int,
	alpha anyvec.Numeric, a anyvec.Vector, lda int,
	b anyvec.Vector, ldb int, beta anyvec.Numeric, ldc int) {
	alphaFloat := alpha.(float32)
	betaFloat := beta.(float32)
	a32 := a.(*vector32)
	b32 := b.(*vector32)
	if v.Overlaps(a32) || v.Overlaps(b32) {
		panic("invalid overlap")
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
		return v.creator.Handle.blas.Sgemm(tb, ta, n, m, k,
			alphaFloat, b32.buffer, ldb, a32.buffer, lda,
			betaFloat, v.buffer, ldc)
	})
}

func (v *vector32) Gemv(trans bool, m, n int, alpha anyvec.Numeric, a anyvec.Vector, lda int,
	x anyvec.Vector, incx int, beta anyvec.Numeric, incy int) {
	alphaFloat := alpha.(float32)
	betaFloat := beta.(float32)
	x32 := x.(*vector32)
	a32 := a.(*vector32)
	if v.Overlaps(x32) || v.Overlaps(a32) {
		panic("invalid overlap")
	}
	v.run(func() error {
		if err := lazyInitAll(true, v, x32, a32); err != nil {
			return err
		}
		tA := cublas.Trans
		if trans {
			tA = cublas.NoTrans
		}
		return v.creator.Handle.blas.Sgemv(tA, n, m, alphaFloat,
			a32.buffer, lda, x32.buffer, incx,
			betaFloat, v.buffer, incy)
	})
}

func (v *vector32) axpy(scaler float32, v1 *vector32) {
	v.assertCompat(v1, false)
	v.run(func() error {
		if v1.buffer == nil {
			return nil
		} else if v.buffer == nil {
			if err := v.lazyInit(false); err != nil {
				return err
			}
			if err := cuda.CopyBuffer(v.buffer, v1.buffer); err != nil {
				return err
			}
			if scaler == 1 {
				return nil
			}
			return v.creator.Handle.blas.Sscal(v.Len(), scaler, v.buffer, 1)
		}
		return v.creator.Handle.blas.Saxpy(v.Len(), scaler, v1.buffer, 1,
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

func (v *vector32) assertCompat(v1 *vector32, readOnly bool) {
	if !readOnly && v.Overlaps(v1) {
		panic("invalid overlap")
	} else if v.Len() != v1.Len() {
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
