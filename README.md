# svd
singular value decomposition for complex matrices

[![GoDoc](https://godoc.org/github.com/ktye/svd?status.svg)](https://godoc.org/github.com/ktye/svd)

The package implements the singular value decomposition (SVD) for general complex matrices in the go programming language.
It has been adapted from the Fortran source referenced below.

Besides the decomposition itself, it can be used to compute the condition of a matrix.

The package is self-contained and uses only the standard library.

## Algorithm
Toms358

## Reference
> Peter Businger, Gene Golub,
> Algorithm 358:
> Singular Value Decomposition of a Complex Matrix,
> Communications of the ACM,
> Volume 12, Number 10, October 1969, pages 564-565.

## Fortran source
http://www.scs.fsu.edu/~burkardt/f77_src/toms358/toms358.f
