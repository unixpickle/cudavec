package cudavec

var testingHandle *Handle

type Fataler interface {
	Fatal(x ...interface{})
}

func setupTest(f Fataler) *Handle {
	if testingHandle != nil {
		return testingHandle
	}
	var err error
	testingHandle, err = NewHandleDefault()
	if err != nil {
		f.Fatal(err)
	}
	return testingHandle
}
