/**
 * @file update_span_impl.hpp
 * @author Chenzhe Diao
 *
 * Implementation of the update method for FrankWolfe algorithm, 
 * recalculate the optimal in the span of previous solution space. 
 * Used as UpdateRuleType in FrankWolfe class. This class is used to 
 * solve l0 optimization, such as OMP method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_UPDATE_SPAN_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_UPDATE_SPAN_IMPL_HPP

// In case it hasn't been included yet.
#include "update_span.hpp"

namespace mlpack {
namespace optimization {

template<typename FunctionType>
void UpdateSpan<FunctionType>::Update(
    const arma::mat& old_coords,
    const arma::mat& s,
    arma::mat& new_coords,
    const size_t num_iter)
{
    // add atom to the solution space here.
    arma::uvec ind = find(s, 1);
    arma::uword d = ind(0);
    AddAtom(d);

    arma::vec b = function.Vectorb();
    arma::mat x = solve(atoms_current, b);
    new_coords = RecoverVector(x);
    
    if (isPrune)
    {
        double eta = 0.5;
        double F0 = function.Evaluate(old_coords);
        double F1 = function.EvaluateFunc(x, atoms_current, b);
        double F = eta*F0 + (1-eta)*F1;
        PruneSupport(F, x);
        new_coords = RecoverVector(x);
    }

}

template<typename FunctionType>
void UpdateSpan<FunctionType>::AddAtom(
    const arma::uword k)
{
    if (isEmpty){
        CurrentIndices() = k;
        CurrentAtoms() = (function.MatrixA()).col(k);
        isEmpty = false;
    }
    else{
        arma::uvec vk(1);
        vk = k;
        current_indices.insert_rows(0, vk);
        
        arma::mat atom = (function.MatrixA()).col(k);
        atoms_current.insert_cols(0, atom);
    }
}

template<typename FunctionType>
arma::vec UpdateSpan<FunctionType>::RecoverVector(
    const arma::vec& x)
{
    int n = (function.MatrixA()).n_cols;
    arma::vec y = arma::zeros<arma::vec>(n);

    arma::uword len = current_indices.size();
    for (size_t ii = 0; ii < len; ++ii)
    {
        y(current_indices(ii)) = x(ii);
    }
    return y;
}

template<typename FunctionType>
void UpdateSpan<FunctionType>::PruneSupport(
    const double F, arma::mat& x)
{
    arma::mat new_atoms = atoms_current;
    arma::uvec new_indices = current_indices;
    arma::vec new_coeff = x;
    arma::vec b = function.Vectorb();

    bool flag = true;
    
    while (flag) {
        // Solve for current gradient
        arma::vec g;
        function.GradientFunc(new_coeff, g, new_atoms, b);
        
        // Find possible atom to be deleted
        arma::vec v = sum(new_atoms % new_atoms, 0);
        v = 0.5*v.t() % new_coeff % new_coeff - new_coeff % g;
        arma::uword ind = v.index_min();
        
        // Try deleting the atom.
        new_atoms.shed_row(ind);
        new_indices.shed_row(ind);
        new_coeff = solve(new_atoms, b);  // recalculate the coeff
        double F_new = function.EvaluateFunc(new_coeff, new_atoms, b);
        
        if (F_new > F)
            // should not delete the atom
            flag = false;
        else {
            // delete the atom from current atoms
            atoms_current = new_atoms;
            current_indices = new_indices;
            x = new_coeff;
        }
    }
    
    
    
}
    
}
}



#endif /* MLPACK_CORE_OPTIMIZERS_FW_UPDATE_SPAN_IMPL_HPP */
