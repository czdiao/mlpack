/**
 * @file frank_wolfe_impl.hpp
 * @author Chenzhe Diao
 *
 * Frank-Wolfe Algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_FRANK_WOLFE_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_FRANK_WOLFE_IMPL_HPP

// In case it hasn't been included yet.
#include "frank_wolfe.hpp"

namespace mlpack {
namespace optimization {

//! Constructor of the FrankWolfe class.
template<
    typename LinearConstrSolverType,
    typename UpdateRuleType>
FrankWolfe<LinearConstrSolverType, UpdateRuleType>::
FrankWolfe(const LinearConstrSolverType linearConstrSolver,
           const UpdateRuleType updateRule,
           const size_t maxIterations,
           const double tolerance) :
    linearConstrSolver(linearConstrSolver),
    updateRule(updateRule),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do*/ }


//! Optimize the function (minimize).
template<
    typename LinearConstrSolverType,
    typename UpdateRuleType>
template<typename FunctionType>
double FrankWolfe<LinearConstrSolverType, UpdateRuleType>::
Optimize(FunctionType& function, arma::mat& iterate)
{
  // To keep track of the function value.
  double currentObjective = function.Evaluate(iterate);

  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat s(iterate.n_rows, iterate.n_cols);
  arma::mat iterateNew(iterate.n_rows, iterate.n_cols);
  double gap = 0;

  for (size_t i = 1; i != maxIterations; ++i)
  {
    // Output current objective function.
    Log::Info << "Iteration " << i << ", objective "
        << currentObjective << "." << std::endl;

    // Calculate the gradient.
    function.Gradient(iterate, gradient);

    // Solve linear constrained problem, solution saved in s.
    linearConstrSolver.Optimize(gradient, s);

    // Check duality gap for return condition.
    gap = std::fabs(dot(iterate - s, gradient));
    if (gap < tolerance)
    {
      Log::Info << "FrankWolfe: minimized within tolerance "
          << tolerance << "; " << "terminating optimization." << std::endl;
      return currentObjective;
    }

    	// Check duality gap for return condition
    	gap = std::fabs(dot(iterate-s, gradient));
    	if (gap < tolerance)
    	{
    		Log::Info << "FrankWolfe: minimized within tolerance "
    				<< tolerance << "; " << "terminating optimization." << std::endl;
    		return CurrentObjective;
    	}

    // Update solution, save in iterateNew.
    updateRule.Update(function, iterate, s, iterateNew, i);

    iterate = std::move(iterateNew);
    currentObjective = function.Evaluate(iterate);
  }

  Log::Info << "Frank Wolfe: maximum iterations (" << maxIterations
      << ") reached; " << "terminating optimization." << std::endl;
  return currentObjective;
}


} // namespace optimization
} // namespace mlpack

#endif
