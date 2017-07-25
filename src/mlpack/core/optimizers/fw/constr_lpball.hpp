/**
 * @file constr_lpball.hpp
 * @author Chenzhe Diao
 *
 * Lp ball constrained for FrankWolfe algorithm. Used as LinearConstrSolverType.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_CONSTR_LPBALL_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_CONSTR_LPBALL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * LinearConstrSolver for FrankWolfe algorithm. Constraint domain given in the
 * form of lp ball. That is, given \f$ v \f$, solve
 * \f[
 * s:=arg\min_{s\in D} <s, v>
 * \f]
 * when \f$ D \f$ is an lp ball.
 *
 * For \f$ p=1 \f$: take (one) \f$ k = arg\max_j |v_j|\f$, then the solution is:
 * \f[
 * s_k = -sign(v_k), \qquad s_j = 0, j\neq k.
 * \f]
 *
 * For \f$ 1<p<\infty \f$: the solution is
 * \f[
 * t_j = -sign(v_j) |v_j|^{p-1}, \qquad s = t/||t||^q
 * \f]
 *
 * For \f$ p=\infty \f$: the solution is
 * \f[
 * s_j = -sign(v_j)
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size.  \f$ i \f$
 * is chosen according to \f$ j \f$ (the iteration number).
 */
class ConstrLpBallSolver
{
 public:
  /**
   * Construct the solver of constrained problem. The constrained domain should
   * be unit lp ball for this class.
   *
   * @param p The constraint is unit lp ball.
   */
  ConstrLpBallSolver(const double p) : p(p)
  { /* Do nothing. */ }
    
    /**
     * Construct the solver of constrained problem, with regularization lambda here.
     *
     * @param p The constraint is unit lp ball.
     * @param lambda Regularization parameter, ideally it should be equal to the lp
     *               norm of each atom.
     */
    ConstrLpBallSolver(const double p, const arma::vec lambda) :
    p(p), reg_flag(true), lambda(lambda)
    { /* Do nothing. */ }


 /**
   * Optimizer of Linear Constrained Problem for FrankWolfe.
   *
   * @param v Input local gradient.
   * @param s Output optimal solution in the constrained domain (lp ball).
   */
  void Optimize(const arma::mat& v,
                arma::mat& s)
  {
    if (p == std::numeric_limits<double>::infinity())
    {
      // l-inf ball.
      s = -sign(v);
      if(reg_flag)
        s = s/lambda;   // element-wise division

      return;
    }
    else if (p > 1.0)
    {
      // lp ball with 1<p<inf.
      if (reg_flag) {
        s = v/lambda;
      }
      else {
        s = v;
      }

      s = -sign(s) % pow(abs(s), p-1);  //element-wise multiplication
      double q = 1/(1.0-1.0/p);
      double qnorm = std::pow(arma::accu(pow(abs(s), q)), 1/q);
      s = s/qnorm;

      if (reg_flag) {
        s = s/lambda;
      }
      return;
    }
    else if (p == 1.0)
    {
      // l1 ball, also used in OMP.
      if (reg_flag) {
        s = arma::abs(v/lambda);
      }
      else {
        s = arma::abs(v);
      }

      arma::uword k;
      s.max(k);  // k is the linear index of the largest element.
      s.zeros();
      s(k) = - mlpack::math::Sign( v(k) );
      if (reg_flag) {
        s = s/lambda;
      }
      return;
    }
    else
    {
      Log::Fatal << "Wrong norm p!" << std::endl;
      return;
    }



  }

  //! Get the p-norm.
  double P() const { return p; }
  //! Modify the p-norm.
  double& P() { return p;}

  //! Get regularization flag
  bool RegFlag() const {return reg_flag;}
  //! Modify regularization flag
  bool& RegFlag() {return reg_flag;}
    
    
 private:
  //! lp norm, 1<=p<=inf;
  //! use std::numeric_limits<double>::infinity() for inf norm.
  double p;

  //! Regularization flag
  bool reg_flag = false;

  //! Regularization parameter, ideally it should be equal to the lp norm of each atom.
  arma::vec lambda;

};

} // namespace optimization
} // namespace mlpack

#endif
