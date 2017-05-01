.PHONY: clean

kernels32.go: kernels32.cu
	nvcc --gpu-architecture=compute_30 --gpu-code=compute_30 --ptx kernels32.cu
	echo 'package cudavec' >$@
	echo '' >>$@
	echo 'var kernels32PTX = `' >>$@
	cat kernels32.ptx | sed -E 's/.version 5\../.version 4.3/' >>$@
	echo '`' >>$@
	rm kernels32.ptx

clean:
	rm kernels32.go
