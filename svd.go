// Package svd implements the singular value decomposition of a complex matrix.
//
// Algorithm
//	Peter Businger, Gene Golub,
//	Algorithm 358:
//	Singular Value Decomposition of a Complex Matrix,
//	Communications of the ACM,
//	Volume 12, Number 10, October 1969, pages 564-565.
//
// Adapted from http://people.sc.fsu.edu/~jburkardt/f77_src/toms358/toms358.f
package svd

import (
	"errors"
	"math"
	"math/cmplx"
)

// Svd is the result of a singular value decomposition A = U*diag(S)*conj(V).
// For a given input matrix A of size m x n, only a compact form is stored.
type Svd struct {
	U [][]complex128 // m x n
	S []float64      // Singular values, length n
	V [][]complex128 // n x n
}

// The original code is more general with mmax and nnmax, defining the leading
// dimensions of A and U (mmax) and V (nmax).
// We restrict the input matrix to the given size: mmax = m, nmax = n.
// The original also defines nu and nv, the number of singular vectors in U and V
// to compute. We set both to n.
// Furthermore, the original defines a parameter p, which is additional data
// in the input matrix A to be transformed. We ignore this and set p = 0.

// New computes the singular value decomposition of A.
// A = U*S*V'.
func New(A [][]complex128) (Svd, error) {
	// Copy matrix A.
	B := make([][]complex128, len(A))
	for i, row := range A {
		B[i] = make([]complex128, len(row))
		copy(B[i], row)
	}

	return NewOverwrite(B)
}

// NewOverwrite is the same as New but overwrites the input matrix.
func NewOverwrite(A [][]complex128) (Svd, error) {
	var b, c, t []float64
	var sn, w, x, y, z, cs, eps, f, g, h float64
	var i, j, k, k1, L, L1 int
	var q complex128
	var U, V [][]complex128
	var S []float64

	m := len(A)
	if m < 1 {
		return Svd{}, errors.New("svd: matrix a has no rows")
	}
	n := len(A[0])
	if n < 1 {
		return Svd{}, errors.New("svd: input has no columns")
	}
	for _, v := range A {
		if len(v) != n {
			return Svd{}, errors.New("svd: input is not a uniform matrix")
		}
	}
	if m < n {
		return Svd{}, errors.New("svd: input matrix has less rows than cols")
	}

	// Allocate temporary and result storage.
	b = make([]float64, n)
	c = make([]float64, n)
	t = make([]float64, n)

	U = make([][]complex128, m)
	for i = range U {
		U[i] = make([]complex128, n)
	}

	S = make([]float64, n)

	V = make([][]complex128, n)
	for i = range V {
		V[i] = make([]complex128, n)
	}

	// Householder Reduction.
	for {
		k1 = k + 1

		// Elimination of A[i][k], i = k, ..., m-1
		z = 0.0
		for i = k; i < m; i++ {
			z += norm(A[i][k])
		}
		b[k] = 0.0
		if z > tol {
			z = math.Sqrt(z)
			b[k] = z
			w = cmplx.Abs(A[k][k])
			q = one
			if w != 0.0 {
				q = A[k][k] / complex(w, 0)
			}
			A[k][k] = q * complex(z+w, 0)
			if k != n-1 {
				for j = k1; j < n; j++ {
					q = zero
					for i = k; i < m; i++ {
						q += cmplx.Conj(A[i][k]) * A[i][j]
					}
					q /= complex(z*(z+w), 0)
					for i = k; i < m; i++ {
						A[i][j] -= q * A[i][k]
					}
				}
			}

			// Phase Transformation.
			q = -cmplx.Conj(A[k][k]) / complex(cmplx.Abs(A[k][k]), 0)
			for j = k1; j < n; j++ {
				A[k][j] *= q
			}
		}

		// Elimination of A[k][j], j=k+2, ..., n-1
		if k == n-1 {
			break
		}
		z = 0.0
		for j = k1; j < n; j++ {
			z += norm(A[k][j])
		}
		c[k1] = 0.0
		if z > tol {
			z = math.Sqrt(z)
			c[k1] = z
			w = cmplx.Abs(A[k][k1])
			q = one
			if w != 0.0 {
				q = A[k][k1] / complex(w, 0)
			}
			A[k][k1] = q * complex(z+w, 0)
			for i = k1; i < m; i++ {
				q = zero
				for j = k1; j < n; j++ {
					q += cmplx.Conj(A[k][j]) * A[i][j]
				}
				q /= complex(z*(z+w), 0)
				for j = k1; j < n; j++ {
					A[i][j] -= q * A[k][j]
				}
			}

			// Phase Transformation.
			q = -cmplx.Conj(A[k][k1]) / complex(cmplx.Abs(A[k][k1]), 0)
			for i = k1; i < m; i++ {
				A[i][k1] *= q
			}
		}
		k = k1
	}

	// Tolerance for negligible elements.
	eps = 0.0
	for k = 0; k < n; k++ {
		S[k] = b[k]
		t[k] = c[k]
		if S[k]+t[k] > eps {
			eps = S[k] + t[k]
		}
	}
	eps *= eta

	// Initialization of U and V.
	for j = 0; j < n; j++ {
		U[j][j] = one
		V[j][j] = one
	}

	// QR Diagonalization.
	for k = n - 1; k >= 0; k-- {

		// Test for split.
		for {
			for L = k; L >= 0; L-- {
				if math.Abs(t[L]) <= eps {
					goto Test
				}
				if math.Abs(S[L-1]) <= eps {
					break
				}
			}

			// Cancellation of E(L)
			cs = 0.0
			sn = 1.0
			L1 = L - 1
			for i = L; i <= k; i++ {
				f = sn * t[i]
				t[i] *= cs
				if math.Abs(f) <= eps {
					goto Test
				}
				h = S[i]
				w = math.Sqrt(f*f + h*h)
				S[i] = w
				cs = h / w
				sn = -f / w
				for j = 0; j < n; j++ {
					x = real(U[j][L1])
					y = real(U[j][i])
					U[j][L1] = complex(x*cs+y*sn, 0)
					U[j][i] = complex(y*cs-x*sn, 0)
				}
			}

			// Test for convergence.
		Test:
			w = S[k]
			if L == k {
				break
			}

			// Origin shift.
			x = S[L]
			y = S[k-1]
			g = t[k-1]
			h = t[k]
			f = ((y-w)*(y+w) + (g-h)*(g+h)) / (2.0 * h * y)
			g = math.Sqrt(f*f + 1.0)
			if f < 0.0 {
				g = -g
			}
			f = ((x-w)*(x+w) + (y/(f+g)-h)*h) / x

			// QR Step.
			cs = 1.0
			sn = 1.0
			L1 = L + 1
			for i = L1; i <= k; i++ {
				g = t[i]
				y = S[i]
				h = sn * g
				g = cs * g
				w = math.Sqrt(h*h + f*f)
				t[i-1] = w
				cs = f / w
				sn = h / w
				f = x*cs + g*sn
				g = g*cs - x*sn
				h = y * sn
				y = y * cs
				for j = 0; j < n; j++ {
					x = real(V[j][i-1])
					w = real(V[j][i])
					V[j][i-1] = complex(x*cs+w*sn, 0)
					V[j][i] = complex(w*cs-x*sn, 0)
				}
				w = math.Sqrt(h*h + f*f)
				S[i-1] = w
				cs = f / w
				sn = h / w
				f = cs*g + sn*y
				x = cs*y - sn*g
				for j = 0; j < n; j++ {
					y = real(U[j][i-1])
					w = real(U[j][i])
					U[j][i-1] = complex(y*cs+w*sn, 0)
					U[j][i] = complex(w*cs-y*sn, 0)
				}
			}
			t[L] = 0.0
			t[k] = f
			S[k] = x
		}

		// Convergence
		if w >= 0.0 {
			continue
		}
		S[k] = -w
		for j = 0; j < n; j++ {
			V[j][k] = -V[j][k]
		}
	}

	// Sort singular values.
	for k = 0; k < n; k++ {
		g = -1.0
		j = k
		for i = k; i < n; i++ {
			if S[i] <= g {
				continue
			}
			g = S[i]
			j = i
		}
		if j == k {
			continue
		}
		S[j] = S[k]
		S[k] = g
		for i = 0; i < n; i++ {
			q = V[i][j]
			V[i][j] = V[i][k]
			V[i][k] = q
		}
		for i = 0; i < n; i++ {
			q = U[i][j]
			U[i][j] = U[i][k]
			U[i][k] = q
		}
	}

	// Back transformation.
	for k = n - 1; k >= 0; k-- {
		if b[k] == 0.0 {
			continue
		}
		q = -A[k][k] / complex(cmplx.Abs(A[k][k]), 0)
		for j = 0; j < n; j++ {
			U[k][j] *= q
		}
		for j = 0; j < n; j++ {
			q = zero
			for i = k; i < m; i++ {
				q += cmplx.Conj(A[i][k]) * U[i][j]
			}
			q /= complex(cmplx.Abs(A[k][k])*b[k], 0)
			for i = k; i < m; i++ {
				U[i][j] -= q * A[i][k]
			}
		}
	}

	if n > 1 {
		for k = n - 2; k >= 0; k-- {
			k1 = k + 1
			if c[k1] == 0.0 {
				continue
			}
			q = -cmplx.Conj(A[k][k1]) / complex(cmplx.Abs(A[k][k1]), 0)
			for j = 0; j < n; j++ {
				V[k1][j] *= q
			}
			for j = 0; j < n; j++ {
				q = zero
				for i = k1; i < n; i++ {
					q += A[k][i] * V[i][j]
				}
				q /= complex(cmplx.Abs(A[k][k1])*c[k1], 0)
				for i = k1; i < n; i++ {
					V[i][j] -= q * cmplx.Conj(A[k][i])
				}
			}
		}
	}

	return Svd{U: U, S: S, V: V}, nil
}

// Condition returns the condition number of the original matrix.
func (s *Svd) Condition() float64 {
	if len(s.S) < 1 {
		return 0
	}
	return s.S[0] / s.S[len(s.S)-1]
}

// The original code is written in single precision with
//	eta = 1.1920929e-07
//	tol = 1.5e-31
// While the original paper uses
//	eta = 1.5E-8
//	tol = 1.E-31
const eta = 2.8E-16  // Relative machine precision. In C this is DBL_EPSILON. // 2.2204460492503131E-16
const tol = 4.0E-293 // The smallest normalized positive number, divided by eta.
// with the smallest normalized positive number: 2.225073858507201e-308 this would be 1.0020841800044862e-292

const zero complex128 = complex(0, 0)
const one complex128 = complex(1, 0)

func norm(z complex128) float64 {
	return real(z)*real(z) + imag(z)*imag(z)
}
