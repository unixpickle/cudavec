package cudavec

import (
	"github.com/unixpickle/anyvec"
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
			0, v.buffer, v.Len())
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

// func (v *vector32) AddChunks(other anyvec.Vector) {
// 	v1 := other.(*vector32)
// 	if v == v1 {
// 		panic("inputs overlap")
// 	} else if v.Len()%v1.Len() != 0 {
// 		panic("scaler count must divide vector size")
// 	}
// 	// TODO: this.
// 	panic("nyi")
// }
//
// func (v *vector32) Rand(p anyvec.ProbDist, r *rand.Rand) {
// 	// TODO: don't forget normal alignment.
// 	panic("nyi")
// }
//
// func (v *vector32) AddRepeated(other anyvec.Vector) {
// 	v1 := other.(*vector32)
// 	if v == v1 {
// 		panic("inputs overlap")
// 	} else if v1.Len() == 0 {
// 		panic("repeated vector cannot be empty")
// 	}
// 	// TODO: this.
// 	panic("nyi")
// }
//
// func (v *vector32) ScaleRepeated(other anyvec.Vector) {
// 	v1 := other.(*vector32)
// 	if v == v1 {
// 		panic("inputs overlap")
// 	} else if v1.Len() == 0 {
// 		panic("repeated vector cannot be empty")
// 	}
// 	// TODO: this.
// 	panic("nyi")
// }
//
// func (v *vector32) AbsSum() anyvec.Numeric {
// 	// TODO: this.
// 	panic("nyi")
// }
//
// func (v *vector32) AbsMax() anyvec.Numeric {
// 	// TODO: this.
// 	panic("nyi")
// }
//
// func (v *vector32) Norm() anyvec.Numeric {
// 	// TODO: this.
// 	panic("nyi")
// }

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
			0, alpha, v.buffer, v.Len())
	})
}

// func (v *vector32) AddLogs(chunkSize int) anyvec.Vector {
// 	if chunkSize < 0 {
// 		panic("chunk size cannot be negative")
// 	} else if chunkSize == 0 {
// 		chunkSize = v.Len()
// 	} else if v.Len()%chunkSize != 0 {
// 		panic("chunk size must divide vector size")
// 	}
// 	if v.Len() == 0 {
// 		return v.creator.MakeVector(0)
// 	}
// 	// TODO: this.
// 	panic("nyi")
// }
//
// func (v *vector32) ElemMax(other anyvec.Vector) {
// 	v1 := other.(*vector32)
// 	v.assertCompat(v1, false)
// 	// TODO: this.
// 	panic("nyi")
// }
//
// func (v *vector32) LogSoftmax(chunkSize int) {
// 	if chunkSize < 0 {
// 		panic("chunk size cannot be negative")
// 	} else if chunkSize == 0 {
// 		chunkSize = v.Len()
// 	} else if v.Len()%chunkSize != 0 {
// 		panic("chunk size must divide vector size")
// 	}
// 	if v.Len() == 0 {
// 		return
// 	}
// 	// TODO: this.
// 	panic("nyi")
// }
//
// func (v *vector32) Pow(n anyvec.Numeric) {
// 	// TODO: this.
// 	panic("nyi")
// }
//
// func (v *vector32) MapMax(cols int) anyvec.Mapper {
// 	if cols < 0 {
// 		panic("column count cannot be negative")
// 	} else if v.Len()%cols != 0 {
// 		panic("column count must divide vector size")
// 	}
// 	if v.Len() == 0 || cols == 0 {
// 		return newMapper32(v.creator, 0, []int{})
// 	}
// 	// TODO: this.
// 	panic("nyi")
// }
//
