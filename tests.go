package cmplxalg

import (
	"fmt"
	"math"
	"math/cmplx"
)

// Testing ---------------------------------------------------------------------

type test struct {
	function testFunc
	name     string
}
type testFunc func() bool

type testCaseSingleMatrix struct {
	a      *Matrix
	result any
}

type testCaseDoubleMatrix struct {
	a, b   *Matrix
	result any
}

type testCaseDoubleVector struct {
	v, w   *Vector
	result any
}

type testCaseSingleVector struct {
	v      *Vector
	result any
}

func TestAll() {
	testMatrices()
	testVectors()
}

func testMatrices() {
	fmt.Println("\nMatrix Tests")
	tests := []test{
		{testMatrixSetGet, "Get/Set"},
		{testMatrixCopy, "Copy"},
		{testZeroes, "Zeroes"},
		{testMatrixMultiplication, "Matrix Multiplication"},
		{testRREF, "Gauss-Jordan (RREF)"},
		{testDeterminant, "Determinant"},
		{testAugment, "Augment"},
		{testInverse, "Inverse"},
	}

	for _, test := range tests {
		fmt.Print("\t" + test.name)
		if test.function() {
			fmt.Println(" OK")
		} else {
			fmt.Println(" failed")
		}
	}
}

func testVectors() {
	fmt.Println("\nVector Tests")
	tests := []test{
		{testParallell, "Parallell"},
		{testPerpendicular, "Perpendicular"},
		{testVectorCopy, "Copy"},
		{testMagnitude, "Magnitude"},
		{testVectorMatrixMultiplication, "Vector*Matrix"},
		{testVectorConjugate, "Conjugate"},
		{testAngle, "Angle"},
		{testAdd, "Addition"},
		{testSub, "Subtraction"},
	}

	for _, test := range tests {
		fmt.Print("\t" + test.name)
		if test.function() {
			fmt.Println(" OK")
		} else {
			fmt.Println(" failed")
		}
	}
}

// Matrix Tests ----------------------------------------------------------------

// TODO: Test matrices of differing amount of rows and columns
func testMatrixSetGet() bool {
	// Testing 'SetRow'
	a := &Matrix{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	}
	r1 := &Vector{1, 2, 3}
	r2 := &Vector{4, 5, 6}
	r3 := &Vector{7, 8, 9}
	a.SetRow(0, r1)
	a.SetRow(1, r2)
	a.SetRow(2, r3)
	expected := &Matrix{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	if !a.Equal(expected) {
		return false
	}

	// Testing 'SetColumn'
	a.SetColumn(0, r1)
	a.SetColumn(2, r3)
	expected = &Matrix{
		{1, 2, 7},
		{2, 5, 8},
		{3, 8, 9},
	}
	if !a.Equal(expected) {
		return false
	}

	// Testing 'GetRow' and 'GetColumn'
	ar := a.GetRow(0)
	ac := a.GetColumn(0)
	a.SetColumn(0, ar)
	a.SetColumn(2, ac)
	expected = &Matrix{
		{1, 2, 1},
		{2, 5, 2},
		{7, 8, 3},
	}
	if !a.Equal(expected) {
		return false
	}

	// None failed, test successful
	return true
}

func testMatrixCopy() bool {
	a := &Matrix{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	b := a
	c := a.Copy()
	a.SetRow(0, &Vector{0, 0, 0})
	return b.Equal(a) && !c.Equal(a)
}

func testZeroes() bool {
	r := 1
	c := 4
	expected := &Matrix{
		{0, 0, 0, 0}}
	if !Zeroes(r, c).Equal(expected) {
		return false
	}

	r = 2
	c = 3
	expected = &Matrix{
		{0, 0, 0},
		{0, 0, 0}}
	if !Zeroes(r, c).Equal(expected) {
		return false
	}
	return true
}

// TODO: Implement complex cases
func testMatrixMultiplication() bool {
	test_cases := []testCaseDoubleMatrix{
		{
			&Matrix{
				{1, 2},
				{-3, -4},
			},
			&Matrix{
				{5, 6},
				{7, 8},
			},
			&Matrix{
				{19, 22},
				{-43, -50},
			},
		},
	}

	for _, test_case := range test_cases {
		test_case.a.MatrixMultiply(test_case.b)
		if !test_case.a.Equal(test_case.result.(*Matrix)) {
			return false
		}
	}
	return true
}

// TODO: Add complex cases
func testRREF() bool {
	test_cases := []testCaseSingleMatrix{
		{
			&Matrix{
				{1, 1},
				{0, 1}},
			&Matrix{
				{1, 0},
				{0, 1}},
		},
		{
			&Matrix{
				{0, 1, 2, 1},
				{1, 1, 3, 4}},
			&Matrix{
				{1, 0, 1, 3},
				{0, 1, 2, 1}},
		},
	}

	for _, test_case := range test_cases {
		test_case.a.RREF()
		if !test_case.a.Equal(test_case.result.(*Matrix)) {
			return false
		}
	}
	return true
}

func testDeterminant() bool {
	test_cases := []testCaseSingleMatrix{
		{
			&Matrix{
				{2, 1},
				{4, 3}},
			2 + 0i,
		},
		{
			&Matrix{
				{0, 1, 2, 1},
				{4, 3, 9, 3},
				{9, 1, 3, 1},
				{1, 1, 3, 4}},
			69 + 0i,
		},
	}

	tolerance := 0.00005
	for _, test_case := range test_cases {
		difference := test_case.a.Determinant() - test_case.result.(complex128)
		if cmplx.Abs(difference) > tolerance {
			return false
		}
	}
	return true
}

func testAugment() bool {
	test_cases := []testCaseDoubleMatrix{
		{
			&Matrix{
				{1, 2},
				{3, 4}},
			&Matrix{
				{4, 5, 6, 7},
				{8, 9, 10, 11}},
			&Matrix{
				{1, 2, 4, 5, 6, 7},
				{3, 4, 8, 9, 10, 11}},
		},
		{
			&Matrix{
				{1},
				{2},
				{3}},
			&Matrix{
				{4},
				{5},
				{6}},
			&Matrix{
				{1, 4},
				{2, 5},
				{3, 6}},
		},
	}

	for _, test_case := range test_cases {
		test_case.a.Augment(test_case.b)
		if !test_case.a.Equal(test_case.result.(*Matrix)) {
			return false
		}
	}
	return true
}

func testInverse() bool {
	test_cases := []testCaseSingleMatrix{
		{
			&Matrix{
				{1, 2, 3},
				{4, 0, 8},
				{3, 0, 5}},
			&Matrix{
				{0, -1.25, 2},
				{0.5, -0.5, 0.5},
				{0, 0.75, -1}},
		},
	}

	for _, test_case := range test_cases {
		test_case.a.Inverse()
		if !test_case.a.Equal(test_case.result.(*Matrix)) {
			return false
		}
	}
	return true
}

// Vector Tests ----------------------------------------------------------------

func testAdd() bool {
	test_cases_add := []testCaseDoubleVector{
		{&Vector{1, 0, 0, 0}, &Vector{-1, 0, 0, 0}, &Vector{0, 0, 0, 0}},
		{&Vector{2, 0, 2, 0}, &Vector{4, 0, 4, 0}, &Vector{6, 0, 6, 0}},
		{&Vector{1i, 0, 0, 0}, &Vector{0, 1i, 0, 0}, &Vector{1i, 1i, 0, 0}},
	}
	for _, test_case := range test_cases_add {
		test_case.v.Add(test_case.w)
		if !test_case.v.Equal(test_case.result.(*Vector)) {
			return false
		}
	}
	return true
}

func testSub() bool {
	test_cases_sub := []testCaseDoubleVector{
		{&Vector{1, 0, 0, 0}, &Vector{-1, 0, 0, 0}, &Vector{2, 0, 0, 0}},
		{&Vector{2, 0, 2, 0}, &Vector{4, 0, 4, 0}, &Vector{-2, 0, -2, 0}},
		{&Vector{1i, 0, 0, 0}, &Vector{0, 1i, 0, 0}, &Vector{1i, -1i, 0, 0}},
	}
	for _, test_case := range test_cases_sub {
		test_case.v.Sub(test_case.w)
		if !test_case.v.Equal(test_case.result.(*Vector)) {
			return false
		}
	}
	return true
}

// TODO: Test complex case
func testAngle() bool {
	test_cases := []testCaseDoubleVector{
		{&Vector{1, 0, 0, 0}, &Vector{-1, 0, 0, 0}, complex128(math.Pi)},
		{&Vector{2, 0, 2, 0}, &Vector{4, 0, 4, 0}, complex128(0)},
		{&Vector{1i, 0, 0, 0}, &Vector{0, 1i, 0, 0}, complex128(math.Pi / 2)},
	}
	margin_of_error := 0.0000001
	for _, test_case := range test_cases {
		angle := Angle(test_case.v, test_case.w)
		difference := angle - test_case.result.(complex128)
		if cmplx.Abs(difference) > margin_of_error {
			return false
		}
	}
	return true
}

func testVectorConjugate() bool {
	test_cases := []testCaseSingleVector{
		{&Vector{1, 2, 3}, &Vector{1, 2, 3}},
		{&Vector{1i, 2i, 3i}, &Vector{-1i, -2i, -3i}},
		{&Vector{0 + 3i, 1 + 2i, 2 + 1i, 3 + 0i}, &Vector{0 - 3i, 1 - 2i, 2 - 1i, 3 - 0i}},
		{&Vector{0}, &Vector{0}},
	}

	for _, test_case := range test_cases {
		test_case.v.Conjugate()
		if !test_case.v.Equal(test_case.result.(*Vector)) {
			return false
		}
	}
	return true
}

// TODO: Complex cases
func testParallell() bool {
	test_cases := []testCaseDoubleVector{
		{&Vector{1, 2, 3}, &Vector{1, 2, 3}, true},
		{&Vector{1, 0, 1}, &Vector{-1, 0, -1}, true},
		{&Vector{1, 2, 3, 4, 5}, &Vector{2, 4, 6, 8, 10}, true},
		{&Vector{-1, -2, -3, -4, -5}, &Vector{2, 4, 6, 8, 10}, true},
		{&Vector{1, 0, 1}, &Vector{1, 0, -1}, false},
		{&Vector{1, 0, 0}, &Vector{0, 0, 1}, false},
		{&Vector{1, 2, 3, 4, 5}, &Vector{-2, 4, -6, 8, -10}, false},
	}

	for _, test_case := range test_cases {
		if test_case.result != Parallell(test_case.v, test_case.w) {
			return false
		}
	}
	return true
}

func testMagnitude() bool {
	test_cases := []testCaseSingleVector{
		{&Vector{0}, 0 + 0i},
		{&Vector{1, 0, 0}, 1 + 0i},
		{&Vector{-2, 0, 0}, 2 + 0i},
		{&Vector{-2, 2, -2, 2}, 4 + 0i},
	}

	for _, test_case := range test_cases {
		if test_case.result != test_case.v.Magnitude() {
			return false
		}
	}
	return true
}

// TODO: Complex cases
func testPerpendicular() bool {
	test_cases := []testCaseDoubleVector{
		{&Vector{1, 0, 1}, &Vector{0, 1, 0}, true},
		{&Vector{1, 1, 0}, &Vector{1, -1, 0}, true},
		{&Vector{1, 1, 0}, &Vector{1, 2, 0}, false},
	}

	for _, test_case := range test_cases {
		if test_case.result != Perpendicular(test_case.v, test_case.w) {
			return false
		}
	}
	return true
}

// TODO: Implement
func testCrossp() bool {
	return false
}

// TODO: Implement
func testDotp() bool {
	return false
}

// TODO: Implement
func testGramSchmidt() bool {
	return false
}

func testVectorCopy() bool {
	v := &Vector{1, 2, 3}
	w := v        // w should be the same as v
	u := v.Copy() // u is a copy of v at this moment
	v.ScalarMultiply(2)
	return w.Equal(v) && !u.Equal(v)
}

// TODO: Complex cases
func testVectorMatrixMultiplication() bool {
	// 4x4 multiplication
	x := &Vector{1, 2, 3, 4}
	a := &Matrix{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
		{13, 14, 15, 16},
	}
	x.MatrixMultiply(a)
	expected := &Vector{90, 100, 110, 120}
	if !x.Equal(expected) {
		return false
	}

	// Complex multiplication
	return true
}
