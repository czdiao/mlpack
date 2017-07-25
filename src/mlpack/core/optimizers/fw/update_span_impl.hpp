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
    atoms.AddAtom(s);

    arma::vec b = function.Vectorb();
    atoms.CurrentCoeffs() = solve(function.MatrixA()*atoms.CurrentAtoms(), b);
    atoms.RecoverVector(new_coords);
}  // Update()
    
}  // namespace optimization
}  // namespace mlpack



#endif /* MLPACK_CORE_OPTIMIZERS_FW_UPDATE_SPAN_IMPL_HPP */
