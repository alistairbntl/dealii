// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2015 by the deal.II authors
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


#include <deal.II/base/geometry_info.h>
#include <deal.II/base/polynomials_macroStokes.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/quadrature_lib.h>
#include <iostream>
#include <iomanip>

DEAL_II_NAMESPACE_OPEN


template <int dim>
PolynomialsMacroStokes<dim>::PolynomialsMacroStokes (const unsigned int k)
  :
  monomials(3),
  n_pols(16),
  p_values(16)
{
  switch (dim)
    {
    case 2:
      for (unsigned int i=0; i<monomials.size(); ++i)
        monomials[i] = Polynomials::Monomial<double> (i);
      break;
    default:
      Assert(false, ExcNotImplemented());
    }
}

template <int dim>
void
PolynomialsMacroStokes<dim>::compute (const Point<dim>            &unit_point,
                              std::vector<Tensor<1,dim> > &values,
                              std::vector<Tensor<2,dim> > &grads,
                              std::vector<Tensor<3,dim> > &grad_grads,
                              std::vector<Tensor<4,dim> > &third_derivatives,
                              std::vector<Tensor<5,dim> > &fourth_derivatives) const
{
  Assert(values.size()==n_pols || values.size()==0,
         ExcDimensionMismatch(values.size(), n_pols));
  Assert(grads.size()==n_pols|| grads.size()==0,
         ExcDimensionMismatch(grads.size(), n_pols));
  Assert(grad_grads.size()==n_pols|| grad_grads.size()==0,
         ExcDimensionMismatch(grad_grads.size(), n_pols));
  Assert(third_derivatives.size()==n_pols|| third_derivatives.size()==0,
         ExcDimensionMismatch(third_derivatives.size(), n_pols));
  Assert(fourth_derivatives.size()==n_pols|| fourth_derivatives.size()==0,
         ExcDimensionMismatch(fourth_derivatives.size(), n_pols));

  (void)third_derivatives;
  Assert(third_derivatives.size()==0,
         ExcNotImplemented());
  (void)fourth_derivatives;
  Assert(fourth_derivatives.size()==0,
         ExcNotImplemented());

  const unsigned int n_sub = 16;
 
  // compute values of polynomials and insert into tensors
  std::vector<std::vector<double> > monovalL(dim, std::vector<double>(4));

  for (unsigned int d=0; d<dim; ++d) {
     monomials[1].value(unit_point(d),monovalL[d]);
  }  

  unsigned int region = 0;
  region = quad_region(unit_point);

  double x = monovalL[0][0];
  double y = monovalL[1][0];
  double xy = monovalL[0][0]*monovalL[1][0];
  double x2 = monovalL[0][0]*monovalL[0][0];
  double y2 = monovalL[1][0]*monovalL[1][0];

  switch (region) {
  case 1:
    values[0][0] = 1.0-1.0*x-1.0*y; 
    values[0][1] = -1.0+1.0*x+1.0*y;
    values[1][0] = 0.5-1.0*y-0.5*x2+0.5*y2; 
    values[1][1] = -1.0*x+1.0*x2+1.0*xy; 
    values[2][0] = +2.0*y-2.0*xy-2.0*y2; 
    values[2][1] = -1.0+2.0*x-1.0*x2+1.0*y2; 
    values[3][0] = -1.0+1.0*x+1.0*y; 
    values[3][1] = 1.0-1.0*x-1.0*y; 
    values[4][0] = -0.5+1.0*y+0.5*x2-0.5*y2; 
    values[4][1] = 1.0*x-1.0*x2-1.0*xy; 
    values[5][0] = -2.0*y+2.0*xy+2.0*y2; 
    values[5][1] = 1.0-2.0*x+1.0*x2-1.0*y2; 
    values[6][0] = 1.0; 
    values[6][1] = 0.; 
    values[7][0] = 1.0*x; 
    values[7][1] = 0.; 
    values[8][0] = 1.0*y; 
    values[8][1] = 0.; 
    values[9][0] = 1.0*y2; 
    values[9][1] = 0.; 
    values[10][0] = 0.; 
    values[10][1] = 1.0; 
    values[11][0] = 0.; 
    values[11][1] = 1.0*x; 
    values[12][0] = 0.; 
    values[12][1] = 1.0*y; 
    values[13][0] = 0.; 
    values[13][1] = +1.0*x2; 
    values[14][0] = -0.5*x2; 
    values[14][1] = 1.0*xy; 
    values[15][0] = -2.0*xy; 
    values[15][1] = 1.0*y2; 
   break;
  case 2:
    values[0][0] = 1.0-1.0*x-1.0*y; 
    values[0][1] = -1.0+1.0*x+1.0*y; 
    values[1][0] = 0.5-1.0*y-0.5*x2+0.5*y2; 
    values[1][1] = -1.0*x+1.0*x2+1.0*xy; 
    values[2][0] = +2.0*y-2.0*xy-2.0*y2; 
    values[2][1] = -1.0+2.0*x-1.0*x2+1.0*y2; 
    values[3][0] = -1.0+2.0*y; 
    values[3][1] = 1.0-2.0*x; 
    values[4][0] = -0.5+1.0*y; 
    values[4][1] = +1.0*x-2.0*x2; 
    values[5][0] = -2.0*y+4.0*y2; 
    values[5][1] = 1.0-2.0*x; 
    values[6][0] = 1.0; 
    values[6][1] = 0.; 
    values[7][0] = +1.0*x; 
    values[7][1] = 0.; 
    values[8][0] = +1.0*y; 
    values[8][1] = 0.; 
    values[9][0] = +1.0*y2; 
    values[9][1] = 0.; 
    values[10][0] = 0.; 
    values[10][1] = 1.0; 
    values[11][0] = 0.; 
    values[11][1] = +1.0*x; 
    values[12][0] = +1.0*x-1.0*y; 
    values[12][1] = +1.0*x; 
    values[13][0] = 0.; 
    values[13][1] = +1.0*x2; 
    values[14][0] = -0.5*y2; 
    values[14][1] = +1.0*x2; 
    values[15][0] = -2.0*y2; 
    values[15][1] = +1.0*x2; 
    break;
  case 3:
    values[0][0] = 0.; 
    values[0][1] = 0.; 
    values[1][0] = 0.; 
    values[1][1] = 0.; 
    values[2][0] = 0.; 
    values[2][1] = 0.; 
    values[3][0] = -1.0*x+1.0*y; 
    values[3][1] = -1.0*x+1.0*y; 
    values[4][0] = -0.5*x2+0.5*y2; 
    values[4][1] = -1.0*x2+1.0*xy; 
    values[5][0] = -2.0*xy+2.0*y2; 
    values[5][1] = -1.0*x2+1.0*y2; 
    values[6][0] = 1.0; 
    values[6][1] = 0.; 
    values[7][0] = +1.0*x; 
    values[7][1] = 0.; 
    values[8][0] = +1.0*y; 
    values[8][1] = 0.; 
    values[9][0] = +1.0*y2; 
    values[9][1] = 0.; 
    values[10][0] = 0.; 
    values[10][1] = 1.0; 
    values[11][0] = 0.; 
    values[11][1] = +1.0*x; 
    values[12][0] = +1.0*x-1.0*y; 
    values[12][1] = +1.0*x; 
    values[13][0] = 0.; 
    values[13][1] = +1.0*x2; 
    values[14][0] = -0.5*y2; 
    values[14][1] = +1.0*x2; 
    values[15][0] = -2.0*y2; 
    values[15][1] = +1.0*x2; 
    break;
  case 4:
    values[0][0] = 0.; 
    values[0][1] = 0.; 
    values[1][0] = 0.; 
    values[1][1] = 0.; 
    values[2][0] = 0.; 
    values[2][1] = 0.; 
    values[3][0] = 0.; 
    values[3][1] = 0.; 
    values[4][0] = 0.; 
    values[4][1] = 0.; 
    values[5][0] = 0.; 
    values[5][1] = 0.; 
    values[6][0] = 1.0; 
    values[6][1] = 0.; 
    values[7][0] = +1.0*x; 
    values[7][1] = 0.; 
    values[8][0] = +1.0*y; 
    values[8][1] = 0.; 
    values[9][0] = +1.0*y2; 
    values[9][1] = 0.; 
    values[10][0] = 0.; 
    values[10][1] = 1.0; 
    values[11][0] = 0.; 
    values[11][1] = +1.0*x; 
    values[12][0] = 0.; 
    values[12][1] = +1.0*y; 
    values[13][0] = 0.; 
    values[13][1] = +1.0*x2; 
    values[14][0] = -0.5*x2; 
    values[14][1] = +1.0*xy; 
    values[15][0] = -2.0*xy; 
    values[15][1] = +1.0*y2; 
    break;
  default:
    Assert(false,ExcNotImplemented());
  }
}


template <int dim>
unsigned int
PolynomialsMacroStokes<dim>::compute_n_pols(unsigned int degree)
{
  if (dim == 2) return 16;
  Assert(false, ExcNotImplemented());
  return 0;
}

template <int dim>
unsigned int
PolynomialsMacroStokes<dim>::quad_region(const Point<dim> &unit_point) const
{
  
  // Change assert message
  Assert(unit_point[0]<=1,ExcNotImplemented());
  Assert(unit_point[1]<=1,ExcNotImplemented());
  Assert(unit_point[0]>=0,ExcNotImplemented());
  Assert(unit_point[1]>=0,ExcNotImplemented());

  if (unit_point[0] >= 0.5 & (1-unit_point[0]) <= unit_point[1] &
      unit_point[1] <= unit_point[0] ) 
    return 1;
  else if (unit_point[1] >= 0.5 & (1-unit_point[1]) <= unit_point[0] &
           unit_point[0] <= unit_point[1] )
    return 2;
  else if (unit_point[0] <= 0.5 & (1-unit_point[0]) >= unit_point[1] &
           unit_point[1] >= unit_point[0] )
    return 3;
  else if (unit_point[1] <= 0.5 & (1-unit_point[0]) >= unit_point[1] &
	   unit_point[1] <= unit_point[0] )
    return 4;

  return 0;
}

template class PolynomialsMacroStokes<2>;


DEAL_II_NAMESPACE_CLOSE
