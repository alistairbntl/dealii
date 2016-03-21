// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2015 by the deal.II authors
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



// Show the shape functions of the macro-const div element on the unit cell
// Plots are gnuplot compatible if lines with desired prefix are selected.

#include "../tests.h"
#include <deal.II/base/logstream.h>
#include <deal.II/fe/fe_macroStokes.h>

#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#define PRECISION 8


template<int dim>
inline void
plot_shape_functions(const unsigned int degree)
{
  FE_MacroStokes<dim> fe_macro_stokes(degree);
  deallog.push(fe_macro_stokes.get_name());

  const unsigned int div=2;
  for (unsigned int my=0; my<=((dim>1) ? div : 0) ; ++my)
      {
        for (unsigned int mx=0; mx<=div; ++mx)
          {
            const Point<dim> p = Point<dim>(1.*mx/div, 1.*my/div);
            deallog << "value " << p;
            for (unsigned int i=0; i<fe_macro_stokes.dofs_per_cell; ++i)
              {
                for (unsigned int c=0; c<dim; ++c)
                  deallog << " " << fe_macro_stokes.shape_value_component(i,p,c);
                deallog << "  ";
              }
            deallog << std::endl;
          }
      }

  deallog.pop();
}


int
main()
{
  std::ofstream logfile ("output");
  deallog << std::setprecision(PRECISION);
  deallog << std::fixed;
  deallog.attach(logfile);
  deallog.threshold_double(1.e-10);

  plot_shape_functions<2>(2);

  return 0;
}
