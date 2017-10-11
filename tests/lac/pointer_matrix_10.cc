// ---------------------------------------------------------------------
//
// Copyright (C) 2006 - 2015 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

// check PointerMatrix:checkAssign

#include "../tests.h"
#include <deal.II/lac/pointer_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

template <typename number>
void
checkAssign(FullMatrix<number> &A, FullMatrix<number> &B)
{
  deallog << "=" << std::endl;
  deallog << "Init with matrix 1" << std::endl;

  PointerMatrix<FullMatrix<number>, Vector<number> > P(&A);

  deallog << "Multiplying with all ones vector" << std::endl;
  Vector<number> V(A.n());
  for (unsigned int i = 0; i < V.size(); ++i)
    V(i) = 1;

  Vector<number> O(A.m());
  P.vmult(O, V);

  // Check the dimensions of the result vector
  Assert(A.m() == O.size(), ExcInternalError());
  deallog << "Dimensions of result vector verified" << std::endl;

  // Verifying results with Method 2: O=A*V
  Vector<number> O_(A.m());
  A.vmult(O_, V);

  Assert(O == O_, ExcInternalError());
  deallog << "Result vector data verified" << std::endl;

  for (unsigned int i = 0; i < O.size(); ++i)
    deallog << O(i) << '\t';
  deallog << std::endl;

  deallog << "Clearing pointer matrix" << std::endl;
  P.clear();

  deallog << "Is matrix empty:" << P.empty() << std::endl;

  deallog << "Assigning pointer matrix to matrix 2" << std::endl;

  P = &B;

  deallog << "Multiplying with all ones vector" << std::endl;
  Vector<number> V_(B.n());
  for (unsigned int i = 0; i < V_.size(); ++i)
    V_(i) = 1;

  Vector<number> OU(B.m());
  P.vmult(OU, V_);

  // Check the dimensions of the result vector
  Assert(B.m() == OU.size(), ExcInternalError());
  deallog << "Dimensions of result vector verified" << std::endl;

  // Verifying results with Method 2: O=B*V
  Vector<number> OU_(B.m());
  B.vmult(OU_, V_);

  Assert(OU == OU_, ExcInternalError());
  deallog << "Result vector data verified" << std::endl;

  for (unsigned int i = 0; i < OU.size(); ++i)
    deallog << OU(i) << '\t';
  deallog << std::endl;
}

int
main()
{

  std::ofstream logfile("output");
  deallog << std::fixed;
  deallog << std::setprecision(4);
  deallog.attach(logfile);

  const double Adata[] =
  { 2, 3, 4, 5 };

  const double Bdata[] =
  { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

  FullMatrix<double> A(2, 2);
  A.fill(Adata);
  FullMatrix<double> B(3, 3);
  B.fill(Bdata);

  checkAssign(A, B);
}
