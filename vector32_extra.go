package cudavec

import (
	"math/rand"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

func (v *vector32) Exp() {
	v.unaryOp("expElements")
}

func (v *vector32) Log() {
	v.unaryOp("logElements")
}

func (v *vector32) Tanh() {
	v.unaryOp("tanhElements")
}

func (v *vector32) Sin() {
	v.unaryOp("sinElements")
}

func (v *vector32) Sigmoid() {
	v.unaryOp("sigmoidElements")
}

func (v *vector32) ClipPos() {
	v.unaryOp("clipPositive")
}

func (v *vector32) unaryOp(kernel string) {
	v.run(func() error {
		if err := v.lazyInit(true); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch(kernel, grid, 1, 1, block, 1, 1,
			0, nil, v.buffer, v.Len())
	})
}

func (v *vector32) Sum() anyvec.Numeric {
	ones := v.Creator().MakeVector(v.Len())
	ones.AddScaler(float32(1))
	return v.Dot(ones)
}

func (v *vector32) ScaleChunks(other anyvec.Vector) {
	v1 := other.(*vector32)
	if v == v1 {
		panic("inputs overlap")
	} else if v.Len()%v1.Len() != 0 {
		panic("scaler count must divide vector size")
	}
	v.run(func() error {
		if err := lazyInitAll(true, v, v1); err != nil {
			return err
		}
		rows := v.Len() / v1.Len()
		cols := v1.Len()
		return v.creator.Handle.blas.Sdgmm(cublas.Right, rows, cols, v.buffer, rows,
			v1.buffer, 1, v.buffer, rows)
	})
}

func (v *vector32) AddChunks(other anyvec.Vector) {
	v1 := other.(*vector32)
	if v == v1 {
		panic("inputs overlap")
	} else if v.Len()%v1.Len() != 0 {
		panic("scaler count must divide vector size")
	}
	v.run(func() error {
		if err := lazyInitAll(true, v, v1); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch("addChunks", grid, 1, 1, block, 1, 1,
			0, nil, v.buffer, v1.buffer, v.Len(), v.Len()/v1.Len())
	})
}

func (v *vector32) Rand(p anyvec.ProbDist, r *rand.Rand) {
	switch p {
	case anyvec.Uniform:
		v.randUniform()
	case anyvec.Bernoulli:
		v.randBernoulli()
	case anyvec.Normal:
		v.randNormal()
	default:
		panic("unsupported distribution")
	}
}

func (v *vector32) randUniform() {
	v.run(func() error {
		if err := v.lazyInit(false); err != nil {
			return err
		}
		if err := v.creator.Handle.gen.Uniform(v.buffer); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch("shiftRandUniform", grid, 1, 1,
			block, 1, 1, 0, nil, v.buffer, v.Len())
	})
}

func (v *vector32) randBernoulli() {
	v.run(func() error {
		if err := v.lazyInit(false); err != nil {
			return err
		}
		if err := v.creator.Handle.gen.Uniform(v.buffer); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch("uniformToBernoulli", grid, 1, 1,
			block, 1, 1, 0, nil, v.buffer, v.Len())
	})
}

func (v *vector32) randNormal() {
	v.run(func() error {
		if err := v.lazyInit(false); err != nil {
			return err
		}
		if v.Len()%2 == 0 {
			return v.creator.Handle.gen.Normal(v.buffer, 0, 1)
		}
		tempBuf, err := cuda.AllocBuffer(v.creator.Handle.allocator, v.buffer.Size()+4)
		if err != nil {
			return err
		}
		if err := v.creator.Handle.gen.Normal(tempBuf, 0, 1); err != nil {
			return err
		}
		return cuda.CopyBuffer(v.buffer, tempBuf)
	})
}

func (v *vector32) AddRepeated(other anyvec.Vector) {
	v.repeatedOp("addRepeated", other.(*vector32))
}

func (v *vector32) ScaleRepeated(other anyvec.Vector) {
	v.repeatedOp("scaleRepeated", other.(*vector32))
}

func (v *vector32) repeatedOp(kernel string, v1 *vector32) {
	if v == v1 {
		panic("inputs overlap")
	} else if v1.Len() == 0 {
		panic("repeated vector cannot be empty")
	}
	v.run(func() error {
		if err := lazyInitAll(true, v, v1); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		if isPowerOf2(v1.Len()) {
			kernel += "Pow2"
			return v.creator.Handle.kernels32.Launch(kernel, grid, 1, 1, block, 1, 1,
				0, nil, v.buffer, v1.buffer, v.Len(), v1.Len()-1)
		} else {
			return v.creator.Handle.kernels32.Launch(kernel, grid, 1, 1, block, 1, 1,
				0, nil, v.buffer, v1.buffer, v.Len(), v1.Len())
		}
	})
}

func (v *vector32) AbsSum() anyvec.Numeric {
	return v.norm(v.creator.Handle.blas.Sasum)
}

func (v *vector32) AbsMax() anyvec.Numeric {
	var res float32
	v.runSync(func() error {
		if v.buffer == nil || v.Len() == 0 {
			return nil
		}
		var idx int
		err := v.creator.Handle.blas.Isamax(v.Len(), v.buffer, 1, &idx)
		if err != nil {
			return err
		}

		outSlice := make([]float32, 1)
		inSlice := cuda.Slice(v.buffer, uintptr(idx-1)*4, uintptr(idx)*4)

		err = cuda.ReadBuffer(outSlice, inSlice)
		res = outSlice[0]
		return err
	})
	if res < 0 {
		res = -res
	}
	return res
}

func (v *vector32) Norm() anyvec.Numeric {
	return v.norm(v.creator.Handle.blas.Snrm2)
}

func (v *vector32) norm(f func(int, cuda.Buffer, int, interface{}) error) anyvec.Numeric {
	var res float32
	v.runSync(func() error {
		if v.buffer == nil {
			return nil
		}
		return f(v.Len(), v.buffer, 1, &res)
	})
	return res
}

func (v *vector32) LessThan(n anyvec.Numeric) {
	v.compare("lessThan", n.(float32))
}

func (v *vector32) GreaterThan(n anyvec.Numeric) {
	v.compare("greaterThan", n.(float32))
}

func (v *vector32) EqualTo(n anyvec.Numeric) {
	v.compare("equalTo", n.(float32))
}

func (v *vector32) compare(kernel string, alpha float32) {
	v.run(func() error {
		if err := v.lazyInit(true); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch(kernel, grid, 1, 1, block, 1, 1,
			0, nil, alpha, v.buffer, v.Len())
	})
}

func (v *vector32) AddLogs(chunkSize int) anyvec.Vector {
	if chunkSize < 0 {
		panic("chunk size cannot be negative")
	} else if chunkSize == 0 {
		chunkSize = v.Len()
	} else if v.Len()%chunkSize != 0 {
		panic("chunk size must divide vector size")
	}
	if v.Len() == 0 {
		return v.creator.MakeVector(0)
	}

	res := v.creator.MakeVector(v.Len() / chunkSize).(*vector32)

	v.run(func() error {
		if err := lazyInitAll(true, v, res); err != nil {
			return err
		}
		err := v.addLogs(v.Len()/chunkSize, chunkSize, res.buffer, v.buffer)
		if err != nil {
			return err
		}
		return nil
	})

	return res
}

func (v *vector32) addLogs(rows, cols int, dst, src cuda.Buffer) error {
	threads := 256
	for threads/2 >= cols && threads > 32 {
		threads /= 2
	}

	for cols > threads {
		dstCols := cols / threads
		if cols%threads != 0 {
			dstCols++
		}
		dstSize := uintptr(dstCols) * uintptr(rows) * 4
		tmp, err := cuda.AllocBuffer(v.creator.Handle.allocator, dstSize)
		if err != nil {
			return err
		}
		if err := v.addLogsKernel(rows, cols, tmp, src, threads); err != nil {
			return err
		}
		src = tmp
		cols = dstCols
	}

	return v.addLogsKernel(rows, cols, dst, src, threads)
}

func (v *vector32) addLogsKernel(rows, cols int, dst, src cuda.Buffer, threads int) error {
	grid := uint(cols / threads)
	if cols%threads != 0 {
		grid++
	}
	sharedSize := 4 * uint(threads)
	return v.creator.Handle.kernels32.Launch("addLogs", uint(rows), grid, 1,
		uint(threads), 1, 1, sharedSize, nil, dst, src, uint(cols))
}

func (v *vector32) ElemMax(other anyvec.Vector) {
	v1 := other.(*vector32)
	v.assertCompat(v1, false)
	v.run(func() error {
		if err := lazyInitAll(true, v, v1); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch("elemMax", grid, 1, 1, block, 1, 1,
			0, nil, v.buffer, v1.buffer, v.Len())
	})
}

func (v *vector32) LogSoftmax(chunkSize int) {
	if chunkSize < 0 {
		panic("chunk size cannot be negative")
	} else if chunkSize == 0 {
		chunkSize = v.Len()
	} else if v.Len()%chunkSize != 0 {
		panic("chunk size must divide vector size")
	}
	if v.Len() == 0 {
		return
	}
	v.run(func() error {
		if err := v.lazyInit(true); err != nil {
			return err
		}
		size := uintptr(v.Len()/chunkSize) * 4
		tmp, err := cuda.AllocBuffer(v.creator.Handle.allocator, size)
		if err != nil {
			return err
		}
		if err := v.addLogs(v.Len()/chunkSize, chunkSize, tmp, v.buffer); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch("subChunks", grid, 1, 1,
			block, 1, 1, 0, nil, v.buffer, tmp, v.Len(), chunkSize)
	})
}

func (v *vector32) Pow(n anyvec.Numeric) {
	scaler := n.(float32)
	v.run(func() error {
		if scaler > 0 && v.buffer == nil {
			return nil
		}
		if err := v.lazyInit(true); err != nil {
			return err
		}
		grid, block := v.kernelSizes()
		return v.creator.Handle.kernels32.Launch("powScaler", grid, 1, 1,
			block, 1, 1, 0, nil, scaler, v.buffer, v.Len())
	})
}

func (v *vector32) MapMax(cols int) anyvec.Mapper {
	if cols < 0 {
		panic("column count cannot be negative")
	} else if v.Len()%cols != 0 {
		panic("column count must divide vector size")
	}
	if v.Len() == 0 {
		return newMapper32(v.creator, 0, []int{})
	}
	rows := v.Len() / cols
	res := &mapper32{creator: v.creator, inSize: v.Len(), outSize: rows}
	v.run(func() error {
		if err := v.lazyInit(true); err != nil {
			return err
		}
		buf, err := cuda.AllocBuffer(v.creator.Handle.allocator, uintptr(rows)*4)
		if err != nil {
			return err
		}
		res.table = buf
		dummyVec := &vector32{size: rows}
		grid, block := dummyVec.kernelSizes()
		return v.creator.Handle.kernels32.Launch("mapMax", grid, 1, 1, block, 1, 1,
			0, nil, buf, v.buffer, rows, cols)
	})
	return res
}

func (v *vector32) SumRows(cols int) anyvec.Vector {
	if cols < 0 {
		panic("column count cannot be negative")
	} else if v.Len()%cols != 0 {
		panic("column count must divide vector size")
	}
	if v.Len() == 0 {
		return v.Creator().MakeVector(cols)
	}
	rows := v.Len() / cols
	res := &vector32{creator: v.creator, size: cols}
	v.run(func() error {
		if err := lazyInitAll(true, v, res); err != nil {
			return err
		}
		ones, err := cuda.AllocBuffer(v.creator.Handle.allocator, uintptr(rows)*4)
		if err != nil {
			return err
		}
		dummy := vector32{size: rows}
		grid, block := dummy.kernelSizes()
		err = v.creator.Handle.kernels32.Launch("setScaler", grid, 1, 1,
			block, 1, 1, 0, nil, float32(1), ones, rows)
		if err != nil {
			return err
		}
		return v.creator.Handle.blas.Sgemm(cublas.NoTrans, cublas.NoTrans,
			cols, 1, rows,
			float32(1),
			v.buffer, cols,
			ones, rows,
			float32(0),
			res.buffer, cols)
	})
	return res
}

func (v *vector32) BatchedGemm(transA, transB bool, num, m, n, k int, alpha anyvec.Numeric,
	a, b anyvec.Vector, beta anyvec.Numeric) {
	a32 := a.(*vector32)
	b32 := b.(*vector32)
	alpha32 := alpha.(float32)
	beta32 := beta.(float32)
	if a32 == v || b32 == v {
		panic("vectors cannot be equal")
	}
	v.creator.run(func() error {
		if err := lazyInitAll(true, a32, b32, v); err != nil {
			return err
		}
		aBatch := a32.splitBatch(num)
		bBatch := b32.splitBatch(num)
		cBatch := v.splitBatch(num)

		for i, subC := range cBatch {
			stream, err := cuda.NewStream(false)
			if err != nil {
				return err
			}
			defer func() {
				stream.Synchronize()
				stream.Close()
			}()

			lda, ldb := k, n
			if transA {
				lda = m
			}
			if transB {
				ldb = k
			}

			tA, tB := cublas.NoTrans, cublas.NoTrans
			if transA {
				tA = cublas.Trans
			}
			if transB {
				tB = cublas.Trans
			}

			if err := v.creator.Handle.blas.SetStream(stream); err != nil {
				return err
			}
			err = v.creator.Handle.blas.Sgemm(tB, tA,
				n, m, k,
				alpha32,
				bBatch[i], ldb,
				aBatch[i], lda,
				beta32,
				subC, n)
			v.creator.Handle.blas.SetStream(nil)
			if err != nil {
				return err
			}
		}
		return nil
	})
}

func (v *vector32) splitBatch(batchSize int) []cuda.Buffer {
	if v.Len()%batchSize != 0 {
		panic("batch size must divide vector length")
	}
	chunkSize := v.buffer.Size() / uintptr(batchSize)
	var res []cuda.Buffer
	for i := 0; i < batchSize; i++ {
		res = append(res, cuda.Slice(v.buffer, uintptr(i)*chunkSize,
			uintptr(i+1)*chunkSize))
	}
	return res
}

func isPowerOf2(n int) bool {
	log := uint(0)
	newNum := n
	for newNum > 1 {
		newNum >>= 1
		log++
	}
	return newNum<<log == n
}
