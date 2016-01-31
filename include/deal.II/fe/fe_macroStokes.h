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

#ifndef dealii__fe_MacroStokes_h
#define dealii__fe_MacroStokes_h

#include <deal.II/base/config.h>
#include <deal.II/base/table.h>
#include <deal.II/base/polynomials_macroStokes.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_poly_tensor.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * The Macro Stokes Element.
 *
**/
template <int dim>
class FE_MarcoStokes
  :
  public FE_PolyTensor<PolynomialsMacroStokes<dim>, dim>
{
public:
  /**
   * Constructor for the MacroStokes element (degree 2).
   */
  FE_PolyTensor ();

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_MacroStokes</tt>
   */
  virtual std::string get_name () const;

  virtual FiniteElement<dim> *clone () const;

  virtual void interpolate(std::vector<double>                &local_dofs,
                           const std::vector<double> &values) const;
  virtual void interpolate(std::vector<double>                &local_dofs,
                           const std::vector<Vector<double> > &values,
                           unsigned int offset = 0) const;
  virtual void interpolate(
    std::vector<double> &local_dofs,
    const VectorSlice<const std::vector<std::vector<double> > > &values) const;
private:
  /**
   * Only for internal use. Its full name is @p get_dofs_per_object_vector
   * function and it creates the @p dofs_per_object vector that is needed
   * within the constructor to be passed to the constructor of @p
   * FiniteElementData.
   */
  static std::vector<unsigned int>
  get_dpo_vector (const unsigned int degree);

  /**
   * Compute the vector used for the @p restriction_is_additive field passed
   * to the base class's constructor.
   */
  static std::vector<bool>
  get_ria_vector (const unsigned int degree);
  /**
   * Initialize the FiniteElement<dim>::generalized_support_points and
   * FiniteElement<dim>::generalized_face_support_points fields. Called from
   * the constructor. See the
   * @ref GlossGeneralizedSupport "glossary entry on generalized support points"
   * for more information.
   */
  void initialize_support_points (const unsigned int bdm_degree);
  /**
   * The values in the face support points of the polynomials needed as
   * test functions. The outer vector is indexed by quadrature points, the
   * inner by the test function. The test function space is PolynomialsP<dim-1>.
   */
  std::vector<std::vector<double> > test_values_face;
  /**
   * The values in the interior support points of the polynomials needed as
   * test functions. The outer vector is indexed by quadrature points, the
   * inner by the test function. The test function space is PolynomialsP<dim>.
   */
  std::vector<std::vector<double> > test_values_cell;
};

DEAL_II_NAMESPACE_CLOSE

#endif
