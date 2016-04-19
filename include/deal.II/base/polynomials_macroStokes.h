// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2016 by the deal.II authors
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

#ifndef dealii__polynomials_MacroStokes_h
#define dealii__polynomials_MacroStokes_h


#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/table.h>
#include <deal.II/base/thread_management.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * This polynomial class is used to build a 
 * vector-valued finite element that produces a point-wise divergence free
 * vector field for Stokes and Navier-Stokes type problems. 
 * The concept is similar to Scott-Vergalius elements on triangles.
 * These functions are composed of continuous quadratic
 * polynomials on four distinct triangular regions with a 
 * constant divergence over the entire quadrilateral element. 
 * Currently, these elements are only implemented in two dimensions.
 *
 * TODO - add grads and gradgrads.  3D?
 * @ingroup Polynomials
 * @author Alistair Bentley / Timo Heister
 * @date 2016
 */
template <int dim>
class PolynomialsMacroStokes
{
public:
  /**
   * Constructor. Creates all basis functions for MacroStokes Element.
   * This element is only valid for degree k = 2
   */
  PolynomialsMacroStokes (const unsigned int k);

  /**
   * Computes the value of each MacroStokes polynomial at @p unit_point.
   * The size of the vectors must either be zero or equal <tt>n()</tt>.  In
   * the first case, the function will not compute these values.
   *
   */
  void compute (const Point<dim>            &unit_point,
                std::vector<Tensor<1,dim> > &values,
                std::vector<Tensor<2,dim> > &grads,
                std::vector<Tensor<3,dim> > &grad_grads,
                std::vector<Tensor<4,dim> > &third_derivatives,
                std::vector<Tensor<5,dim> > &fourth_derivatives) const;

  /**
   * Returns the number of basis elements for the MacroElement.  (Note: This
   * class will always return 16.)
   */
  unsigned int n () const;

  /**
   * Returns the degree of the MacroStokes polynomials resticted to each 
     triangular sub-element (Note: This will always return 2)
   */
  unsigned int degree () const;

  /**
   * Return the name of the space, which is <tt>MacroStokes</tt>.
   */
  std::string name () const;

  /**
   * Return the number of polynomials in the space <tt>MacroStokes(2)</tt>
   * without requiring to build a MacroStokes object. This is required
   * by the FiniteElement classes.
   */
  static unsigned int compute_n_pols(unsigned int degree);


protected:

  /**
   * Given a point on the reference element, this function returns
   * the macro element region which is used to select the correct
   * polynomial for evaluation.
   */
  unsigned int quad_region(const Point<dim> &unit_point) const;


private:

  /**
   * Storage for monomials. In 2D, this is just the polynomial of order
   * <i>k</i>. No 3D implmentation.
   */
  std::vector<Polynomials::Polynomial<double> > monomials;

  /**
   * There are 16 MacroStokes polynomials.
   */
  unsigned int n_pols;

  /**
   * A mutex that guards the following scratch arrays.
   */
  mutable Threads::Mutex mutex;

  /**
   * Auxiliary memory.
   */
  mutable std::vector<double> p_values;

  /**
   * Auxiliary memory.
   */
  //  mutable std::vector<Tensor<1,dim> > p_grads;

  /**
   * Auxiliary memory.
   */
  //  mutable std::vector<Tensor<2,dim> > p_grad_grads;

  /**
   * Auxiliary memory.
   */
  //  mutable std::vector<Tensor<3,dim> > p_third_derivatives;
  /**
   * Auxiliary memory.
   */
  //  mutable std::vector<Tensor<4,dim> > p_fourth_derivatives;
};


template <int dim>
inline unsigned int
PolynomialsMacroStokes<dim>::n() const
{
  return 16;
}


template <int dim>
inline unsigned int
PolynomialsMacroStokes<dim>::degree() const
{
  return 2;
}


template <int dim>
inline std::string
PolynomialsMacroStokes<dim>::name() const
{
  return "MacroStokes";
}


DEAL_II_NAMESPACE_CLOSE

#endif
