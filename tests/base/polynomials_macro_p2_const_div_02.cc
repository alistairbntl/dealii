// ---------------------------------------------------------------------
//
// Copyright (C) 2010 - 2015 by the deal.II authors
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


// plot Polynomials macro_divFree on reference element

#include "../tests.h"
#include <deal.II/base/logstream.h>
#include <deal.II/base/job_identifier.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/polynomials_macroStokes.h>
#include <deal.II/base/quadrature_lib.h>

#include <vector>
#include <iomanip>
#include <fstream>

using namespace std;

template <int dim>
void plot(const PolynomialsMacroStokes<dim> &poly)
{
  QTrapez<1> base_quadrature;
  QIterated<dim> quadrature(base_quadrature, poly.degree()+1);
  std::vector<Tensor<1,dim> > values(poly.n());
  std::vector<Tensor<2,dim> > grads;
  std::vector<Tensor<3,dim> > grads2;
  std::vector<Tensor<4,dim> > thirds;
  std::vector<Tensor<5,dim> > fourths;

  for (unsigned int k=0; k<quadrature.size(); ++k)
    {

      deallog << "MacroConstDiv" << poly.degree() << '<' << dim << '>'
              << '\t' << quadrature.point(k) << '\t'  << std::endl;

      poly.compute(quadrature.point(k), values, grads, grads2, thirds, fourths);

      for (unsigned int i=0; i<poly.n(); ++i)
        for (unsigned int d=0; d<dim; ++d)
          deallog << '\t' << values[i][d];
      deallog << std::endl;
    }
}

int main()
{
  const std::string logname = "output";
  std::ofstream logfile(logname.c_str());
  deallog << std::setprecision(3);
  deallog.attach(logfile);
  deallog.threshold_double(1.e-10);

  PolynomialsMacroStokes<2> p20(2) ;

  plot(p20);
}
