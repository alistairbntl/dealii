// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2016 by the deal.II authors
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


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/polynomials_p.h>
#include <deal.II/base/table.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_macroStokes.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>

#include <iostream>
#include <sstream>


DEAL_II_NAMESPACE_OPEN

template <int dim>
FE_MacroStokes<dim>::FE_MacroStokes (const unsigned int deg)
  :
  FE_PolyTensor<PolynomialsMacroStokes<dim>, dim> (
    deg,
    FiniteElementData<dim>(get_dpo_vector(),
                           dim, deg+1, FiniteElementData<dim>::Hdiv),
    get_ria_vector (deg),
    std::vector<ComponentMask>(PolynomialsMacroStokes<dim>::compute_n_pols(deg),
                               std::vector<bool>(dim,true)))
{

  Assert (dim == 2, ExcImpossibleInDim(dim));
  Assert (deg == 2, ExcMessage("MacroStokes Elements are only implemented for degree 2"));

  const unsigned int n_dofs = this->dofs_per_cell;

  this->mapping_type = mapping_bdm;
  // TODO -- this should probably be mapping_piola

  // Set up the generalized support
  // points
  initialize_support_points (deg);
  FullMatrix<double> M(n_dofs, n_dofs);
  FETools::compute_node_matrix(M, *this);

  //  std::cout << std::endl;
  //  M.print_formatted(std::cout, 2, true);

  this->inverse_node_matrix.reinit(n_dofs, n_dofs);
  this->inverse_node_matrix.invert(M);
}



template <int dim>
std::string
FE_MacroStokes<dim>::get_name () const
{
  // note that the
  // FETools::get_fe_from_name
  // function depends on the
  // particular format of the string
  // this function returns, so they
  // have to be kept in synch

  std::ostringstream namebuf;
  namebuf << "MacroStokes<";

  return namebuf.str();
}


template <int dim>
FiniteElement<dim> *
FE_MacroStokes<dim>::clone() const
{
  return new FE_MacroStokes<dim>(*this);
}



template <int dim>
void
FE_MacroStokes<dim>::interpolate(
  std::vector<double> &,
  const std::vector<double> &) const
{
  Assert(false, ExcNotImplemented());
}


template <int dim>
void
FE_MacroStokes<dim>::interpolate(
  std::vector<double> &,
  const std::vector<Vector<double> > &,
  unsigned int) const
{
  Assert(false, ExcNotImplemented());
}



template <int dim>
void
FE_MacroStokes<dim>::interpolate(
  std::vector<double> &local_dofs,
  const VectorSlice<const std::vector<std::vector<double> > > &values) const
{
  AssertDimension (values.size(), dim);
  Assert (values[0].size() == this->generalized_support_points.size(),
          ExcDimensionMismatch(values.size(), this->generalized_support_points.size()));
  Assert (local_dofs.size() == this->dofs_per_cell,
          ExcDimensionMismatch(local_dofs.size(),this->dofs_per_cell));
  
  // First evaluate dofs at vertices.
  unsigned int dbase = 0;
  for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      local_dofs[dbase+2*v] = values[0][v];
      local_dofs[dbase+2*v+1] = values[1][v];
    }
  
  dbase += 2*GeometryInfo<dim>::vertices_per_cell;
  // Second, calculate line integrals around edges

  // We need edge points with weights for the integrals
  QGauss<dim-1> edge_points (3);
  Quadrature<dim> edges = QProjector<dim>::project_to_all_faces(edge_points);
  unsigned int num_vertices = GeometryInfo<dim>::vertices_per_cell;

  for (unsigned int e = 0; e < GeometryInfo<dim>::faces_per_cell; ++e)
  {
    double s0 = 0;
    double s1 = 0;
    for (unsigned int k = 0; k < 3; ++k)
    {
      s0 +=  values[0][3*e + num_vertices + k] * edges.weight(3*e + k) ;
      s1 +=  values[1][3*e + num_vertices + k] * edges.weight(3*e + k) ;
    }
    local_dofs[dbase+2*e] = s0;
    local_dofs[dbase+2*e + 1] = s1;
  }

  dbase += 2*GeometryInfo<dim>::faces_per_cell;

  Assert (dbase == this->dofs_per_cell, ExcInternalError());
}




template <int dim>
std::vector<unsigned int>
FE_MacroStokes<dim>::get_dpo_vector ()
{
  // Since this element is only designed for degree 2 the
  // degrees of freedom are not dynamic
  unsigned int vertex_dofs = 2;
  unsigned int edge_dofs = 2;
  unsigned int interior_dofs = 0;
  
  std::vector<unsigned int> dpo(dim+1);
  dpo[0] = vertex_dofs;
  dpo[dim-1] = edge_dofs;
  dpo[dim]   = interior_dofs;

  return dpo;
}



template <int dim>
std::vector<bool>
FE_MacroStokes<dim>::get_ria_vector (const unsigned int deg)
{
  if (dim==1)
    {
      Assert (false, ExcImpossibleInDim(1));
      return std::vector<bool>();
    }
  if (dim==3)
    {
      Assert(false, ExcImpossibleInDim(3));
      return std::vector<bool>();
    }

  const unsigned int dofs_per_cell = PolynomialsMacroStokes<dim>::compute_n_pols(deg);

  std::vector<bool> ret_val(dofs_per_cell,true);

  return ret_val;
}

template <int dim>
void
FE_MacroStokes<dim>::initialize_support_points (const unsigned int deg)
{
  // Support points are vertices and quadrature points on the
  // edges of the cell.  One the edges, we need to calculate the line
  // integrals which need to be exact for quadratics.
  this->generalized_support_points.resize(GeometryInfo<dim>::vertices_per_cell
				  + (deg+1)*GeometryInfo<dim>::faces_per_cell);
  // Generalized support points for vertices
  for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
  {
    this->generalized_support_points[v] = GeometryInfo<dim>::unit_cell_vertex(v);
  }
  // The next set of support points are for line integrals on
  // the edges.  Since the functions are quadratics on the boundary
  // we use 3 Gaussian points for the quadrature.
  QGauss<dim-1> edge_points (deg + 1);
  Quadrature<dim> edges = QProjector<dim>::project_to_all_faces(edge_points);

  for (unsigned int k=0; k < edges.size(); ++k)
    this->generalized_support_points[k+GeometryInfo<dim>::vertices_per_cell] 
      = edges.point(k);
}



/*-------------- Explicit Instantiations -------------------------------*/
template class FE_MacroStokes<2>;

DEAL_II_NAMESPACE_CLOSE

