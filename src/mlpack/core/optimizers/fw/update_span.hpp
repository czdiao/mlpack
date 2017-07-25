/**
 * @file update_span.hpp
 * @author Chenzhe Diao
 *
 * Update method for FrankWolfe algorithm, recalculate the optimal in the span
 * of previous solution space. Used as UpdateRuleType.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_UPDATE_SPAN_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_UPDATE_SPAN_HPP

#include <mlpack/prereqs.hpp>
#include "atoms.hpp"

namespace mlpack {
namespace optimization {

/**
 * Recalculate the optimal solution in the span of all previous solution space, 
 * used as update step for FrankWolfe algorithm.
 *
 *
 * For UpdateSpan to work, FunctionType template parameters are required.
 * This class must implement the following functions:
 *
 * FunctionType:
 *
 *   arma::mat MatrixA()
 *   arma::vec Vectorb()
 *   double Evaluate(const arma::mat& coords)
 *   void Gradient(const arma::mat& coords, arma::mat& gradient)
 *   double EvaluateFunc(const arma::mat& coords,
 *              const arma::mat& AA, const arma::vec& bb)
 *   void GradientFunc(const arma::mat& coords, arma::mat& gradient,
 *              const arma::mat& AA, const arma::vec& bb)
 *
 *
 * MatrixA() returns a matrix with all the atoms as its columns.
 * Vectorb() returns a vector we want to approximate with the atoms.
 *
 * PruneSupport is an optional step, which is based on Algorithm 2 of:
 * Rao, Nikhil, Parikshit Shah, and Stephen Wright.
 * "Forward–backward greedy algorithms for atomic norm regularization." 
 * IEEE Transactions on Signal Processing 63.21 (2015): 5798-5811.
 *
 * Notice that this this code solves the unconstraint update
 * (used in l0 optimization such as OMP), while Algorithm 2 in the above paper
 * tries to solve the constraint problem.
 *
 * @tparam FunctionType Objective function type to be minimized in FrankWolfe algorithm.
 */
template<typename FunctionType>
class UpdateSpan
{
 public:
  /**
   * Construct the span update rule. The function to be optimized is 
   * input here.
   *
   * @param function Function to be optimized in FrankWolfe algorithm.
   */
    UpdateSpan(FunctionType& function): function(function)
    { /* Do nothing. */ }

 /**
  * Update rule for FrankWolfe, reoptimize in the span of original solution space.
  * This class also keeps record of all previously used atoms, this function also
  * add a new atom into the record.
  *
  *
  * @param old_coords previous solution coords.
  * @param s current linear_constr_solution result.
  * @param new_coords new output solution coords.
  * @param num_iter current iteration number
  */
  void Update(const arma::mat& old_coords,const arma::mat& s,
	  arma::mat& new_coords, const size_t num_iter);

  //! Get the instantiated function to be optimized.
  FunctionType Function() const { return function; }
  //! Modify the instantiated function.
  FunctionType& Function() { return function; }


 private:
  //! The instantiated function.
  FunctionType& function;
  
  Atoms atoms;
};

} // namespace optimization
} // namespace mlpack

// Include implementation
#include "update_span_impl.hpp"

#endif
