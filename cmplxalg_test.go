package cmplxalg_test

import (
	"reflect"
	"testing"

	"github.com/velosov/cmplxalg"
)

func Test(t *testing.T) {
	e1, e2 := cmplxalg.Vector{1, 0}, cmplxalg.Vector{0, 1}
	gate, I := cmplxalg.Matrix{{0, 0}, {1, 0}}, cmplxalg.Matrix{{1, 0}, {0, 1}}
	E1, E2 := *e1.Copy(), *e1.Copy()

	E1.MatrixMultiply(&I)
	if !reflect.DeepEqual(E1, e1) {
		t.Errorf("Wrong multiplication")
		t.Log(E1, e1)
		return
	}
	t.Log("Identity Successful")

	E2.MatrixMultiply(&gate)
	if !reflect.DeepEqual(E2, e2) {
		t.Errorf("Wrong multiplication")
		t.Log(E2, e2)
		return
	}

	t.Log("Completed")
}
