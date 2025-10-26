package cmplxalg

import (
	"fmt"
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
