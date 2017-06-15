/**
 * @file update_linesearch.hpp
 * @author Chenzhe Diao
 *
 * Minimize convex function with line search, using secant method. 
 * In FrankWolfe algorithm, used as UpdateRuleType.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_UPDATE_LINESEARCH_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_UPDATE_LINESEARCH_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Use line search in the update step for FrankWolfe algorithm. That is, take
 * \f$ \gamma = arg\min_{\gamma\in [0, 1]} f((1-\gamma)x + \gamma s) \f$.
 * The update rule would be:
 * \f[
 * x_{k+1} = (1-\gamma) x_k + \gamma s
 * \f]
 *
 *
 * For UpdateLineSearch to work, FunctionType template parameters are required.
 * This class must implement the following functions:
 *
 * FunctionType:
 *
 *   double Evaluate(const arma::mat& coordinates);
 *          Evaluation of the function at specific coordinate.
 *
 *   void Gradient(const arma::mat& coordinates,
 *                 arma::mat& gradient);
 *          Solve the gradient of the function at specific coordinate, returned in gradient.
 *
 *
 *
 * @tparam FunctionType Objective function to be minimized with line search
 *                      between 2 points.
 */
template <typename FunctionType>
class UpdateLineSearch
{
public:
    /**
     * Construct the line search update rule. The function to be optimized is
     * input here.
     *
     * @param function Function to be optimized.
     * @param maxIter Max number of iterations in line search.
     * @param tolerance Tolerance for termination of line search.
     */
    UpdateLineSearch(FunctionType& function,
                     const size_t maxIter = 100000,
                     const double tolerance = 1e-5):
    function(function),
    maxIter(maxIter),
    tolerance(tolerance)
    {/* Do nothing */}
    

    /**
     * Update rule for FrankWolfe, optimize with line search.
     *
     * @param old_coords One endpoint of line search, previous solution coords in FW.
     * @param s The other endpoint of line search, current linear_constr_solution result in FW.
     * @param new_coords Output optimal solution coords.
     * @param num_iter Current iteration number, not used here.
     */
    void Update(const arma::mat& old_coords,
                const arma::mat& s,
                arma::mat& new_coords,
                const size_t num_iter)
    {
        double gamma = LineSearchSecant(old_coords, s);
        new_coords = old_coords + gamma * deltaX;
    }

    
    //! Get the instantiated function to be optimized.
    const FunctionType& Function() const { return function; }
    //! Modify the instantiated function.
    FunctionType& Function() { return function; }

    //! Get the tolerance for termination.
    double Tolerance() const {return tolerance;}
    //! Modify the tolerance for termination.
    double& Tolerance() {return tolerance;}
    
    //! Get the maximum number of iterations (0 indicates no limit).
    size_t MaxIterations() const { return maxIter; }
    //! Modify the maximum number of iterations (0 indicates no limit).
    size_t& MaxIterations() { return maxIter; }

private:
    //! The instantiated function.
    FunctionType& function;
    
    //! Tolerance for convergence.
    double tolerance;

    //! Max number of iterations.
    size_t maxIter;

    //! Start point of line search.
    arma::mat X0;

    //! (End_point - Start_point) of line search.
    arma::mat deltaX;

    /**
     * Line search to minimize function between two points with Secant method, that is, 
     * to find the zero of Derivative(gamma), where gamma is in [0,1].
     *
     * The function is assumed to be convex here.
     * If the function is convex, Derivative(gamma) would be a nondecreasing function.
     * If the function is strongly convex, Derivative(gamma) is strictly increasing,
     *
     * @param x1 One end point.
     * @param x2 The other end point.
     * @return Optimal solution position ratio betwen two points, 0 means x1, 1 means x2.
     */
    double LineSearchSecant(const arma::mat& x1,
                      const arma::mat& x2)
    {
        // Set up the search line, that is, find the zero of der(gamma) = Derivative(gamma).
        X0 = x1;
        deltaX = x2 - x1;
        double gamma = 0;
        double gamma_new = 0;
        double der = Derivative(0);
        double der_new = Derivative(1);
        double sec = der_new - der ;    // secant, should always >=0 for convex function.
        
        if (der >= 0.0) // optimal sol at left endpoint
            return 0.0;
        else if (der_new <= 0.0) //optimal sol at righ endpoint
            return 1.0;
        else if (sec < tolerance) // function too flat
            return 0.0;
        
        // Line search by Secant Method
        for (size_t k = 0; k<maxIter; ++k) {
            if (sec < 0.0)
            {
                Log::Fatal << "LineSearchSecant: Function is not convex!"  << std::endl;
                return 0.0;
            }
            
            gamma_new = gamma - der / sec;
            gamma_new = std::max(gamma_new, 0.0);
            gamma_new = std::min(gamma_new, 1.0);
            
            der_new = Derivative(gamma_new);
            
            sec = (der_new - der)/(gamma_new - gamma);
            gamma = gamma_new;
            der = der_new;

            if(std::fabs(der) < tolerance)
            {
            	Log::Info << "LineSearchSecant: minimized within tolerance "
            			<< tolerance << "; " << "terminating optimization." << std::endl;
            	return gamma;
            }
        }

        Log::Info << "LineSearchSecant: maximum iterations (" << maxIter
        		<< ") reached; " << "terminating optimization." << std::endl;
        return gamma;
    }
    
    /**
     * Derivative of the function along the search line.
     *
     * @param gamma position of the point in the search line, take in [0, 1].
     */
    double Derivative(const double gamma)
    {
        arma::mat gradient;
        function.Gradient(X0 + gamma*deltaX, gradient);
        return dot(gradient, deltaX);
    }

};
    
    
    
    
} // namespace optimization
} // namespace mlpack

#endif
