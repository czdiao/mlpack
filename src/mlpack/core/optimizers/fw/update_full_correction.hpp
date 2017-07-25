/**
 * @file update_full_correction.hpp
 * @author Chenzhe Diao
 *
 * Update method for FrankWolfe algorithm, recalculate the coefficents of
 * of current atoms, while satisfying the norm constraint.
 * Used as UpdateRuleType.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_UPDATE_FULL_CORRECTION_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_UPDATE_FULL_CORRECTION_HPP

#include <mlpack/prereqs.hpp>
#include "atoms.hpp"

namespace mlpack {
namespace optimization {

template<typename FunctionType>
class UpdateFullCorrection
{
 public:
  UpdateFullCorrection(FunctionType& function,
                       double tau,
                       double stepSize):
    function(function), tau(tau), stepSize(stepSize)
  { /* Do nothing. */ }
  
  /**
   * Update rule for FrankWolfe, recalculate the coefficents of of current
   * atoms, while satisfying the norm constraint.
   *
   * @param old_coords previous solution coords.
   * @param s current linear_constr_solution result.
   * @param new_coords new output solution coords.
   * @param num_iter current iteration number.
   */
  void Update(const arma::mat& old_coords,const arma::mat& s,
              arma::mat& new_coords, const size_t num_iter)
  {
    // Line search.
    arma::mat v = tau * s - old_coords;
    arma::mat b = function.Vectorb();
    arma::mat A = function.MatrixA();
    double gamma = arma::dot(b - A * old_coords, A * v);
    gamma = gamma / std::pow(arma::norm(A * v, "fro"), 2);
    gamma = std::min(gamma, 1);
    atoms.CurrentCoeffs() = (1.0-gamma) * atoms.CurrentCoeffs();
    atoms.AddAtom(s, gamma * tau);

    // Projected gradient method for enhancement
    atoms.ProjectedGradientEnhancement(tau, function, stepSize);
    atoms.RecoverVector(new_coords);
  }

 private:
  FunctionType& function;
  double tau;
  double stepSize;
  Atoms atoms;
};

} // namespace optimization
} // namespace mlpack

#endif
