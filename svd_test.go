package svd

import (
	"fmt"
	"math/cmplx"
	"math/rand"
	"testing"
)

// conj returns the complex conjugate of a matrix.
func conj(A [][]complex128) [][]complex128 {
	m := len(A)
	n := len(A[0])
	B := make([][]complex128, n)
	for i := 0; i < n; i++ {
		B[i] = make([]complex128, m)
		for k := 0; k < m; k++ {
			B[i][k] = cmplx.Conj(A[k][i])
		}
	}
	return B
}

// mul multiplies two matrices and returns the result.
func mul(A [][]complex128, B [][]complex128) [][]complex128 {
	m := len(A)
	n := len(A[0])
	if len(B) != n {
		panic("inner matrix dimensions mismatch")
	}
	l := len(B[0])
	C := make([][]complex128, m)
	for i := range C {
		C[i] = make([]complex128, l)
		for k := 0; k < l; k++ {
			C[i][k] = dot(A, B, i, k, n)
		}
	}
	return C
}

// dot returns the scalar product of two vectors from matrices A[i][*] and B[*][k], with inner dimension n.
func dot(A, B [][]complex128, i, k, n int) complex128 {
	var c complex128
	for j := 0; j < n; j++ {
		c += A[i][j] * B[j][k]
	}
	return c
}

// diag returns a matrix with the given vector as the diagonal and zeros otherwise.
func diag(v []float64) [][]complex128 {
	A := make([][]complex128, len(v))
	for i := range A {
		A[i] = make([]complex128, len(v))
		A[i][i] = complex(v[i], 0)
	}
	return A
}

// conjdot returns the scalar product of two vectors, where the first is conjugated.
func conjdot(a, b []complex128) complex128 {
	var sum complex128
	if len(a) != len(b) {
		panic("conjdot: input vector sizes don't match")
	}
	for i := range a {
		sum += cmplx.Conj(a[i]) * b[i]
	}
	return sum
}

// compare tests if two matrices are similar.
func compare(A, B [][]complex128) (err error, maxerr float64) {
	eps := 1.0E-12
	if len(A) != len(B) {
		return fmt.Errorf("matrix dimensions differ"), 0
	}
	if len(A) < 1 {
		return fmt.Errorf("empty matrix"), 0
	}
	n := len(A[0])
	for i := range A {
		if len(A[i]) != n || len(B[i]) != n {
			return fmt.Errorf("matrix dimensions differ"), 0
		}
		for k := range A[i] {
			if e := cmplx.Abs(A[i][k] - B[i][k]); e > eps {
				return fmt.Errorf("matrix differs by: %v", e), 0
			} else if e > maxerr {
				maxerr = e
			}
		}
	}
	return nil, maxerr
}

// column returns a column from a matrix.
func column(A [][]complex128, col int) []complex128 {
	v := make([]complex128, len(A))
	for i := range A {
		v[i] = A[i][col]
	}
	return v
}

// isunitary tests if a matrix is unitary.
func isunitary(A [][]complex128) (err error, maxerr float64) {
	eps := 1.0E-12
	for i := range A[0] {
		a := column(A, i)
		for k := range A[0] {
			b := column(A, k)
			s := conjdot(a, b)
			target := complex(0.0, 0.0)
			if i == k {
				target = complex(1.0, 0.0)
			}
			if e := cmplx.Abs(s - target); e > eps {
				fmt.Println(i, k, s, target)
				return fmt.Errorf("unitary fails with element difference: %v", e), 0
			} else if e > maxerr {
				maxerr = e
			}
		}
	}
	return nil, maxerr
}

func TestSvd(t *testing.T) {
	// Create a random matrix.
	m := 10
	n := 4
	A := make([][]complex128, m)
	for i := range A {
		A[i] = make([]complex128, n)
		for k := range A[i] {
			A[i][k] = complex(rand.NormFloat64(), rand.NormFloat64())
		}
	}

	// Do the singular value decomposition.
	S, err := New(A)
	if err != nil {
		t.Fatal(err)
	}

	// Compare the original with the product of the resulting matrices.
	// Check if the product is similar to the original.
	E := diag(S.S)
	P := mul(mul(S.U, E), conj(S.V))
	if err, _ := compare(A, P); err != nil {
		t.Fatal(err)
	}

	// Check for unitarity of U.
	if err, _ := isunitary(S.U); err != nil {
		t.Fatal(err)
	}

	// Check for unitarity of V.
	if err, _ := isunitary(S.V); err != nil {
		t.Fatal(err)
	}
}
