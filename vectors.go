package cmplxalg

import "math/cmplx"

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
		(*v)[i] = DotProduct(a.GetRow(i), v_original)
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
