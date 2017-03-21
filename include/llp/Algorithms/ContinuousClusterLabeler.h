#ifndef __ContinuousClusterLabeller_h
#define __ContinuousClusterLabeller_h

#include <unordered_set>

#include "ceres/ceres.h"

#include "bd/BaggedDataset.h"

/*
  Find optimal proportional labelling of K clusters of N Bags.
  Let 
     x \in [0,1]^K
     p \in [0,1]^N
     C \in R^{N \times K}, with sum_j C_{i,j} = 1, forall i.

  x is an unkown labelling of clusters we want to find
  p is a known labelling of bags
  C maps cluster labels to bag labels

  We solve this as a box-constrained nonlinear least squares problem with
  regularization.

  ----------------
  Using CeresCostFunction solves the following
  
  Let
     r_i(x) = (p_i - (C*x)_i), for 1 <= i <= N
     r_{N+1} = sum_i p_i - (C*x)_i
     \lambda >= 0

  r_i(x)     is the signed error on bag i
  r_{N+1}(x) is the sum of signed errors over all bags. 
             The purpose of this term is to make the labelling respect the
	     population label proportions.
  \lambda    is a regularization parameter that determines that determines the
             weighting of local/population error. If set to N, it will weigh the
	     terms equally.

  Then we want to solve

     argmin_x 1/2 sum_i r_i(x)^2 + \lambda r_{N+1}(x)^2
     s.t
       0 <= x_j <= 1, for 1 <= j <= K

  ----------------------------------------
  Using CeresCostFunction2 solves the simpler problem without population constraints
  Let
     r_i(x) = (p_i - (C*x)_i), for 1 <= i <= N

  r_i(x)     is the signed error on bag i

  Then we want to solve

     argmin_x 1/2 sum_i r_i(x)^2
     s.t
       0 <= x_j <= 1, for 1 <= j <= K

*/

template< template<typename,typename> class TCostFunction, size_t BagLabelDim=1 >
class ContinuousClusterLabeler
{
public:
  typedef ContinuousClusterLabeler< TCostFunction, BagLabelDim > Self;
  typedef BaggedDataset< BagLabelDim, 1 > BaggedDatasetType;
  
  typedef typename BaggedDatasetType::MatrixType MatrixType;
  typedef typename BaggedDatasetType::BagLabelVectorType BagLabelVectorType;
  typedef typename BaggedDatasetType::InstanceLabelVectorType ClusterLabelVectorType;

  typedef TCostFunction<BagLabelVectorType, MatrixType> CostFunctionType;



  typedef struct {} ParameterType;
  
  ContinuousClusterLabeler( const ParameterType& params=ParameterType() )
    : m_Params( params )
  {}
  
  ~ContinuousClusterLabeler() {}
    
  /*
    @param bags           Set of bags including their known labels
    @param clusterBagMap  Mapping from cluster labels to bag labels
    @param labeling       Final labeling

    @return   Objective value at best cluster labeling
   */
  double Label( const BaggedDatasetType& bags,
		const MatrixType& clusterBagMap,
		ClusterLabelVectorType& labeling ) {

    // Ceres is responsible for freeing
    CostFunctionType* costFunction = new CostFunctionType(bags.BagLabels(), clusterBagMap);
    
    // Create the parameter group
    std::vector< double* > params;
    params.push_back(labeling.data());

    // Setup the problem
    ceres::Problem problem;
    problem.AddResidualBlock( costFunction, NULL, params );
    for ( int i = 0; i < labeling.size(); ++i ) {
      problem.SetParameterLowerBound( params[0], i, 0);
      problem.SetParameterUpperBound( params[0], i, 1);
    }

    // Solve the problem
    ceres::Solver::Options options;
    options.max_num_iterations = 150;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    if ( summary.IsSolutionUsable() ) {
      return summary.final_cost;
    }

    // Improve on this
    std::cerr << "Solution is not usable!" << std::endl
	      << summary.BriefReport() << std::endl;
    return std::numeric_limits<double>::infinity();      
  }

  
 private:
  ParameterType m_Params;
  
};

#endif
