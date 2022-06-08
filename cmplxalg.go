package cmplxalg

import (
	"fmt"
	"math"
	"math/cmplx"
	"strings"
)

// TO BE IMPLEMENTED -----------------------------------------------------------
/*

Fixes/Misc
- Further complex testing with complex test-cases
- Replace 'len(m)', 'len(m[0])' with 'm.Rows()', 'm.Columns()'
- Replace 'panic' with 'errors' package
- Document time complexity for all methods and functions
- Prevent creation of matrices with uneven row lengths
- Split into seperate files, 'matrix.go', 'vector.go', 'testing.go'
- Test for 1x1 matrices
- Examine float imprecision for RR and RREF
- Decide if methods with needed copying should or should not return new matrix
- Time-complexity for every method/function
- To pad or not to pad? Currently it varies between functions
- Uniformity in comments ending with or without '.'

Tolerance (to combat float imprecision)
- Introduce global tolerance constants
- Use tolerance margins for equality
- Tolerance margins for row reduction

Methods/Functions
- Rank
- Nullspace
- Example uses
- Eigenvectors and values (see https://en.wikipedia.org/wiki/Divide-and-conquer_eigenvalue_algorithm)

*/

// Types -----------------------------------------------------------------------

type Matrix [][]complex128
type Vector []complex128

// Panic Messages --------------------------------------------------------------
const ERR_DOTP_DIM = "Cannot calculate dot-product of vectors with different dimensions"
const ERR_CRSP_DIM = "Can only calculate cross product of 3-dimensional vectors"
const ERR_NORM_ZER = "Cannot normalize the zero-vector"
const ERR_PARA_ZER = "Cannot determine if zero-vector is parallell to another vector"
const ERR_PERP_ZER = "Cannot determine if zero-vector is perpendicular to another vector"
const ERR_MTRX_RNG = "The index to get/set is out of range"
const ERR_MTRX_DIM = "Incompatible matrix dimensions"
const ERR_MTRX_SNG = "Cannot row-reduce singular matrix"
const ERR_AUGM_ROW = "Cannot augment matrices with differing amount of rows"
const ERR_INVR_SQR = "Cannot calculate inverse of non-square matrix"
const ERR_INVR_SNG = "Cannot calculate inverse of singular matrix"
const ERR_DETM_SQR = "Cannot calculate inverse of non-square matrix"

// Matrices --------------------------------------------------------------------

// Return copy of matrix 'm'
// O(n*m)
func (m *Matrix) Copy() *Matrix {
	elements := make([][]complex128, len(*m))
	for i := 0; i < len(*m); i++ {
		elements[i] = make([]complex128, len((*m)[0]))
		copy(elements[i], (*m)[i])
	}

	newMatrix := Matrix(elements)
	return &newMatrix
}

// Get column 'index' of matrix 'm' starting at 0.
// Panics if out of range.
// O(n)
func (m *Matrix) GetColumn(index int) *Vector {
	assertColumnInRange(index, m)
	rows := m.Rows()
	v := new(Vector)
	*v = make([]complex128, rows)
	for i := 0; i < rows; i++ {
		(*v)[i] = (*m)[i][index]
	}
	return v
}

// Get row 'index' of matrix 'm' starting at 0.
// Panics if out of range.
// O(n)
func (m *Matrix) GetRow(index int) *Vector {
	assertRowInRange(index, m)
	columns := m.Columns()
	v := new(Vector)
	*v = make([]complex128, columns)
	for i := 0; i < columns; i++ {
		(*v)[i] = (*m)[index][i]
	}
	return v
}

// Set column 'index' of matrix 'm' to 'col'.
// Column index starts at 0.
// Panics if out of range.
// O(n)
func (m *Matrix) SetColumn(index int, col *Vector) {
	assertColumnInRange(index, m)
	rows := len((*m)[0])
	for i := 0; i < rows; i++ {
		(*m)[i][index] = (*col)[i]
	}
}

// Set column 'index' of matrix 'm' to 'row'.
// Column index starts at 0.
// Panics if out of range.
// O(n)
func (m *Matrix) SetRow(index int, row *Vector) {
	assertRowInRange(index, m)
	cols := len((*m)[0])
	for i := 0; i < cols; i++ {
		(*m)[index][i] = (*row)[i]
	}
}

// Returns matrix with specified size filled with zeroes.
// O(n*m)
// TODO: Panic if rows or columns == 0
func Zeroes(rows, columns int) *Matrix {
	elements := make([][]complex128, rows)
	for i := 0; i < rows; i++ {
		elements[i] = make([]complex128, columns)
	}

	newMatrix := Matrix(elements)
	return &newMatrix
}

// Returns identity matrix of specified size.
// O(n^2)
func Identity(size int) *Matrix {
	m := Zeroes(size, size)
	for i := 0; i < size; i++ {
		(*m)[i][i] = 1
	}
	return m
}

// Multiplies matrix 'm' by matrix 'a'
// O(n^3)
func (m *Matrix) MatrixMultiply(a *Matrix) {
	// Check compatible
	m_cols, m_rows := m.Columns(), m.Rows()
	a_cols, a_rows := a.Columns(), a.Rows()
	if m_cols != a_rows || m_rows != a_cols {
		panic(ERR_MTRX_DIM)
	}

	result := Zeroes(m_rows, a_cols)
	for i := 0; i < m_rows; i++ {
		for j := 0; j < a_cols; j++ {
			(*result)[i][j] = DotProduct(m.GetRow(i), a.GetColumn(j))
		}
	}
	*m = *result
}

// Multiplies each element in matrix 'm' by scalar 'f'
// O(n*m)
func (m *Matrix) ScalarMultiply(f complex128) {
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			(*m)[i][j] *= f
		}
	}
}

// Adds scalar 'f' to each element in matrix 'm'
// O(n*m)
func (m *Matrix) ScalarAdd(f complex128) {
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			(*m)[i][j] += f
		}
	}
}

// Returns amount of rows in matrix 'm'
// O(1)
func (m *Matrix) Rows() int {
	return len(*m)
}

// Returns amount of columns in matrix 'm'
// O(1)
func (m *Matrix) Columns() int {
	return len((*m)[0])
}

// Transposes and conjugates matrix 'm'.
// O(n*m)
func (m *Matrix) HermitianTranspose() {
	m.Transpose()
	m.Conjugate()
}

// Returns whether matrix 'm' is hermitian or not.
// O(n*m)
func (m *Matrix) IsHermitian() bool {
	mH := m.Copy()
	mH.HermitianTranspose()
	return m.Equal(mH)
}

// Conjugates matrix 'm'.
// O(n*m)
func (m *Matrix) Conjugate() {
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			(*m)[i][j] = cmplx.Conj((*m)[i][j])
		}
	}
}

// Transposes matrix 'm'.
// O(n*m)
func (m *Matrix) Transpose() {
	m_rows, m_cols := m.Rows(), m.Columns()
	mT := Zeroes(m_rows, m_cols)
	for i := 0; i < m_rows; i++ {
		for j := 0; j < m_cols; j++ {
			(*mT)[i][j] = (*m)[j][i]
		}
	}
	*m = *mT // TODO: Check performance? Potentially awful
}

// Inverses matrix 'm'.
// Panics if matrix isn't invertible.
func (m *Matrix) Inverse() {
	if m.Columns() != m.Rows() {
		panic(ERR_INVR_SQR)
	}
	if m.IsSingular() {
		panic(ERR_INVR_SNG)
	}

	// Augment identity matrix
	m.Augment(Identity(m.Columns()))
	m.RREF()
	inverse := Zeroes(m.Rows(), m.Columns()/2)
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns()/2; j++ {
			(*inverse)[i][j] = (*m)[i][j+m.Columns()/2]
		}
	}
	*m = *inverse
}

// Augments matrix 'm' with matrix 'a'.
// Panics if sizes are incompatible.
// O(n*m + p*q)
func (m *Matrix) Augment(a *Matrix) {
	if m.Rows() != a.Rows() {
		panic(ERR_AUGM_ROW)
	}
	new_matrix := Zeroes(m.Rows(), m.Columns()+a.Columns())

	// Elements from 'm'
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			(*new_matrix)[i][j] = (*m)[i][j]
		}
	}

	// Elements from 'a'
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < a.Columns(); j++ {
			(*new_matrix)[i][j+m.Columns()] = (*a)[i][j]
		}
	}

	*m = *new_matrix
}

// Returns determinant of square matrix 'm'
// Panics if 'm' is not square.
// TODO: Unsure if works for complex matrices
func (m *Matrix) Determinant() complex128 {
	// Assert square matrix
	if m.Rows() != m.Columns() {
		panic(ERR_DETM_SQR)
	}

	m_c := m.Copy()
	m_c.RR()
	determinant := complex128(-1) // Why is this -1 and not 1???
	for i := 0; i < m.Rows(); i++ {
		determinant *= (*m_c)[i][i]
	}
	return determinant
}

// Returns whether 'm' is invertible or not.
func (m *Matrix) IsInvertible() bool {
	return m.IsSingular()
}

// Returns whether 'm' is singular or not.
// TODO: Defer to recover from singular matrix panic in Determinant()
func (m *Matrix) IsSingular() bool {
	tolerance := 0.0000001
	return cmplx.Abs(m.Determinant()) < tolerance
}

// Perform row reduction on matrix 'm'.
// Panics if singular matrix.
func (m *Matrix) RR() {
	m_rows, m_cols := m.Rows(), m.Columns()
	for k := range *m {
		// Skip overflowing rows
		if k >= m_cols {
			break
		}
		// Find pivot for column k:
		iMax := k
		max := cmplx.Abs((*m)[k][k])
		for i := k + 1; i < m_rows; i++ {
			if abs := cmplx.Abs((*m)[i][k]); abs > max {
				iMax = i
				max = abs
			}
		}
		if (*m)[iMax][k] == 0 {
			// TODO: General Error
			panic(ERR_MTRX_SNG)
		}
		// swap rows(k, i_max)
		(*m)[k], (*m)[iMax] = (*m)[iMax], (*m)[k]
		// Do for all rows below pivot:
		for i := k + 1; i < m_rows; i++ {
			// Do for all remaining elements in current row:
			for j := k + 1; j < m_cols; j++ {
				(*m)[i][j] -= (*m)[k][j] * ((*m)[i][k] / (*m)[k][k])
			}
			// Fill lower triangular matrix with zeros:
			(*m)[i][k] = 0
		}
	}
}

// Find Row-Reduced Echelon Form of matrix 'm'.
// Panics if singular matrix.
func (m *Matrix) RREF() {
	m_rows := m.Rows()

	// Produce over-triangular matrix
	m.RR()

	// Reduce all rows
	for j := 0; j < m_rows; j++ {
		pivot_row := m.GetRow(j)
		pivot, exists := getPivot(pivot_row)
		if !exists {
			break
		}
		for i := 0; i < m_rows; i++ {
			if i == j {
				continue // Skip pivot row
			}
			factor := (*m)[i][j] / pivot
			row := m.GetRow(i)
			to_sub := pivot_row.Copy()
			to_sub.ScalarMultiply(factor)
			row.Sub(to_sub)
			m.SetRow(i, row)
		}

		// Divide to get leading one on pivot row
		pivot_row.ScalarMultiply(1 / pivot)
		m.SetRow(j, pivot_row)
	}
}

// Helper function used in RREF
func getPivot(v *Vector) (complex128, bool) {
	for i := 0; i < len(*v); i++ {
		if (*v)[i] != 0 {
			return (*v)[i], true
		}
	}
	return 0, false
}

// Returns whether 'a' is equal to 'b' by comparing elementwise.
// O(n*m)
func (a *Matrix) Equal(b *Matrix) bool {
	// Get dimensions
	a_rows, b_rows := len(*a), len(*b)
	a_cols, b_cols := len((*a)[0]), len((*b)[0])

	// Check dimensions
	if a_rows != b_rows || a_cols != b_cols {
		return false
	}

	// Check elements
	for i := 0; i < a_rows; i++ {
		for j := 0; j < a_cols; j++ {
			if (*a)[i][j] != (*b)[i][j] {
				return false
			}
		}
	}
	return true
}

// Return string representation of matrix 'm'
// O(n*m)
func (m *Matrix) String() string {
	s := strings.Builder{}
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			s.WriteString(fmt.Sprint((*m)[i][j]))
			s.WriteString("\t")
		}
		s.WriteString("\n")
	}
	return s.String()
}

// Return string representation of real matrix 'm'
// O(n*m)
func (m *Matrix) RealString() string {
	s := strings.Builder{}
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			s.WriteString(fmt.Sprint(real((*m)[i][j])))
			s.WriteString("\t")
		}
		s.WriteString("\n")
	}
	return s.String()
}

// Vectors ---------------------------------------------------------------------

// TODO: Implement padding in all vector-functions.
// Make sure that only the vector whose-method is called
// gets changed.

// Conjugate vector element-wise.
// O(n)
func (v *Vector) Conjugate() {
	for i := 0; i < len(*v); i++ {
		(*v)[i] = cmplx.Conj((*v)[i])
	}
}

// Return magnitude of vector
// O(n)
func (v *Vector) Magnitude() complex128 {
	return cmplx.Sqrt(DotProduct(v, v))
}

// Returns copy of vector.
// O(n)
func (v *Vector) Copy() *Vector {
	elements := make([]complex128, len(*v))
	copy(elements, *v)
	newVector := Vector(elements)
	return &newVector
}

// Add 'w' to vector 'v'. Pads vector with zeroes if different sizes
// O(n), n is highest dimension of 'v', 'w'
func (v *Vector) Add(w *Vector) {
	v_c, w_c := copyAndPad(v, w)
	for i := 0; i < len(*v); i++ {
		(*v_c)[i] += (*w_c)[i]
	}
	*v = *v_c
}

// Subtract 'w' from vector 'v'. Pads vector with zeroes if different sizes
// O(n), n is highest dimension of 'v', 'w'
func (v *Vector) Sub(w *Vector) {
	w_c := w.Copy()
	w_c.ScalarMultiply(-1)
	v.Add(w_c)
}

// Returns dot product 'v*w'.
// Panics if 'v' and 'w' differs in dimension.
// O(n)
func DotProduct(v, w *Vector) complex128 {
	// Check appropriate Magnitude
	vlen := len(*v)
	if vlen != len(*w) {
		panic(ERR_DOTP_DIM)
	}

	// Iterate and sum products
	var dotp complex128
	for i := 0; i < vlen; i++ {
		dotp += (*v)[i] * cmplx.Conj((*w)[i])
	}
	return dotp
}

// Returns cross product 'vxw'.
// Panics if 'v' or 'w' not 3-dimensional.
// O(1)
func CrossProduct(v, w *Vector) *Vector {
	// Check appropriate Magnitude
	vlen := len(*v)
	wlen := len(*w)
	if vlen != 3 || wlen != 3 {
		panic(ERR_CRSP_DIM)
	}

	// Formula for cross-product
	// https://en.wikipedia.org/wiki/Cross_product
	return &Vector{
		(*v)[1]*(*w)[2] - (*v)[2]*(*w)[1],
		(*v)[2]*(*w)[0] - (*v)[0]*(*w)[2],
		(*v)[0]*(*w)[1] - (*v)[1]*(*w)[0]}
}

// Returns whether 'v' and 'w' are perpendicular.
// If 'v' and 'w' differ in dimension it pads the smaller one with zeroes.
// Panics if 'v' or 'w' are zero-vectors
// O(n)
func Parallell(v, w *Vector) bool {
	// Check not-zero vector
	if v.Magnitude() == 0 || w.Magnitude() == 0 {
		panic(ERR_PARA_ZER)
	}

	// Copy and pad to same length
	v, w = copyAndPad(v, w)

	// Normalize
	v.Normalize()
	w.Normalize()

	// Check if same direction
	same := v.Equal(w)

	// Check if opposite direction
	v.ScalarMultiply(-1)
	opposite := v.Equal(w)

	return same || opposite
}

// Normalize 'v'.
// O(n)
func (v *Vector) Normalize() {
	// Check if zero-vector
	magnitude := v.Magnitude()
	if v.Magnitude() == 0 {
		panic(ERR_NORM_ZER)
	}

	// Normalize elements
	for i := 0; i < len(*v); i++ {
		(*v)[i] /= magnitude
	}
}

// Multiply 'v' by complex factor.
// O(n)
func (v *Vector) ScalarMultiply(factor complex128) {
	for i := 0; i < len(*v); i++ {
		(*v)[i] *= factor
	}
}

// Returns whether 'v' and 'w' are perpendicular.
// Panics if 'v' or 'w' are zero-vectors
// O(n)
func Perpendicular(v, w *Vector) bool {
	// Check not-zero vector
	if v.Magnitude() == 0 || w.Magnitude() == 0 {
		panic(ERR_PERP_ZER)
	}

	// Pad and check parallell by use of dot product
	v, w = copyAndPad(v, w)
	return DotProduct(v, w) == 0+0i
}

// Returns the angle between vectors 'v' and 'w'
// O(n)
func Angle(v, w *Vector) complex128 {
	// cos(angle) = v * w / |v||w|
	return cmplx.Acos(DotProduct(v, w) / (v.Magnitude() * w.Magnitude()))
}

// Returns whether 'v' is equal to 'w' by comparing element-wise.
// O(n)
func (v *Vector) Equal(w *Vector) bool {
	if len(*v) != len(*w) {
		return false
	}
	for i, v_element := range *v {
		if v_element != (*w)[i] {
			return false
		}
	}
	return true
}

// Multiply vector 'v' by matrix 'a'.
// O(n)
func (v *Vector) MatrixMultiply(a *Matrix) {
	v_original := v.Copy()
	for i := 0; i < len(*v); i++ {
		(*v)[i] = DotProduct(a.GetColumn(i), v_original)
	}
}

// Functions -------------------------------------------------------------------

// Helper Functions ------------------------------------------------------------

// Return maximum of two ints
func max(x, y int) int {
	if x < y {
		return y
	}
	return x
}

// Copy vectors 'v' and 'w' and pad the shortest one to match the length of the
// other.
func copyAndPad(v, w *Vector) (*Vector, *Vector) {
	// Copy with eventual padding
	v_len, w_len := len(*v), len(*w)
	Magnitude_difference := v_len - w_len
	switch {
	case Magnitude_difference < 0:
		// Pad v
		v_arr := new([]complex128)
		*v_arr = make([]complex128, len(*w))
		v = (*Vector)(v_arr)
		w = w.Copy()
	case Magnitude_difference > 0:
		// Pad w
		w_arr := new([]complex128)
		*w_arr = make([]complex128, len(*v))
		w = (*Vector)(w_arr)
		v = v.Copy()
	default:
		v = v.Copy()
		w = w.Copy()
	}
	return v, w
}

// Assert that 'row' is in range of matrix 'm'.
// Panic otherwise.
func assertRowInRange(row int, m *Matrix) {
	rows := len(*m)
	if row > rows-1 || row < 0 {
		panic(ERR_MTRX_RNG)
	}
}

// Assert that 'column' is in range of matrix 'm'.
// Panic otherwise.
func assertColumnInRange(column int, m *Matrix) {
	columns := len((*m)[0])
	if column > columns-1 || column < 0 {
		panic(ERR_MTRX_RNG)
	}
}

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
