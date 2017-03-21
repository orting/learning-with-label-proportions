#ifndef __CMSTrainer_h
#define __CMSTrainer_h

#include "libcmaes/cmaes.h"

#include "llp/Models/ClusterModel.h"
#include "llp/Algorithms/Trainers/CMSTrainerParameters.h"
#include "bd/BaggedDataset.h"

// TODO: Rewrite to return a model

/**

   TInstanceClusterer should define the types
     ParameterType
     DistanceType
     InstanceClusteringType
   and the methods
     TClusterer( ParameterType& )
     InstanceClusteringType Cluster( BaggedDataset&, const DistanceType& )
     
   TLabeler should define the types
     ParameterType
   and the methods
     TLabeler( ParameterType& )
     double Label( BaggedDataset&, BaggedDataset::MatrixType&, VectorType& )
 
   TTracer should define the types
     ParameterType
   and the methods
     TTracer( const ParameterType& )
     void Trace( double )
*/
template< typename TBaggedDataset,
	  typename TInstanceClusterer,
	  typename TClusterLabeler,
	  typename TTracer >
class CMSTrainer {
public:
  typedef TBaggedDataset BaggedDatasetType;
  typedef typename BaggedDatasetType::InstanceLabelVectorType ClusterLabelVectorType; 
  
  typedef TInstanceClusterer ClustererType;
  typedef TClusterLabeler    LabelerType;
  typedef TTracer            TracerType;
  
  typedef CMSTrainer< BaggedDatasetType, ClustererType, LabelerType, TracerType > Self;

  typedef typename ClustererType::InstanceClusteringType InstanceClusteringType;
  typedef typename ClustererType::DistanceType DistanceType;
  typedef ClusterModel< DistanceType, BaggedDatasetType > ModelType;
  typedef typename ModelType::MatrixType MatrixType;

  typedef CMSTrainerParameters                  ParameterType;
  typedef typename ClustererType::ParameterType ClustererParameterType;
  typedef typename LabelerType::ParameterType   LabelerParameterType;
  typedef typename TracerType::ParameterType    TracerParameterType;

  typedef libcmaes::GenoPheno< libcmaes::pwqBoundStrategy > GenoPheno;
  typedef libcmaes::CMAParameters< GenoPheno > CMAParameters;
  typedef libcmaes::CMASolutions CMASolutions;  

  CMSTrainer( const ParameterType&          trainerParams   = ParameterType(),
	      const ClustererParameterType& clustererParams = ClustererParameterType(),
	      const LabelerParameterType&   labelerParams   = LabelerParameterType(),
	      const TracerParameterType&    tracerParams    = TracerParameterType() )
    : m_Params( trainerParams )
    , m_ClustererParams( clustererParams )
    , m_LabelerParams( labelerParams )
    , m_TracerParams( tracerParams )
    , m_TrainError( std::numeric_limits<double>::infinity() )
  {}

  ~CMSTrainer(){}

  double TrainError() const {
    return m_TrainError;
  }
  
  /**
     \brief Train cluster model         
     \param bags   Training data as a collection of bags
     \param dim    The dimension of the feature space. If this is not equal to 
                   the dimension of bags, then it is assumed that the bags lie 
		   in a an dim x (bags.Dimension()/dim) dimensional feature space
  */
  typename ModelType::Pointer
  Train( BaggedDatasetType& bags, const size_t dim ) {
    // We are searhing for feature weights in [0,1] and we start at the center.
    // It is recomended that sigma is set so the optimal solution is within
    // [0.5 - sigma, 0.5 + sigma].
    std::vector< double > weights( dim, 0.5 );
    
    std::vector< double > lbounds( dim, 0.0 );
    std::vector< double > ubounds( dim, 1.0 );
    GenoPheno gp( &lbounds.front(), &ubounds.front(), dim );
    CMAParameters cmaParams( weights.size(),
			     weights.data(),
			     m_Params.sigma,
			     m_Params.lambda,
			     m_Params.seed,
			     gp );
    cmaParams.set_algo( aCMAES );

    if ( m_Params.maxIterations > 0 ) {
      cmaParams.set_max_iter( m_Params.maxIterations );
    }

    if ( !m_Params.out.empty() ) {
      cmaParams.set_fplot( m_Params.out );
    }


    ClustererType clusterer( m_ClustererParams );
    LabelerType   labeler( m_LabelerParams );
    TracerType    tracer( m_TracerParams );
    
    
    // Define and wrap the objective for cmaes
    std::function< double(const double*, const int&) > objective =
      [&bags, &clusterer, &labeler, &tracer]
      ( const double* w, const int& N )
      {
	DistanceType dist(w, N);
	for ( int i = 0; i < N; ++i ) {
	  tracer.Trace("Weight " + std::to_string(i), w[i]);
	}
	InstanceClusteringType clustering = clusterer.Cluster( bags, dist );
	tracer.Trace("ClusterBagMap", clustering.clusterBagMap );
	
	// In some cases we have a clustering algorithm that is not guaranteed to
	// give us the requested number of clusters, so we need to check how many
	// we actually got
	ClusterLabelVectorType labels = ClusterLabelVectorType::Zero( clustering.NumberOfClusters() );
	double risk = labeler.Label( bags, clustering.clusterBagMap, labels );
	
	tracer.Trace("Risk", risk );
	return risk;
      };
  
    tracer.Info("Status", "Running CMA-ES");

    // Run the optimization
    CMASolutions solutions = libcmaes::cmaes< GenoPheno >( objective, cmaParams );

    // TODO: Handle the diferent ways that CMAES can terminate
    if ( solutions.run_status() < 0 ) {
      tracer.Error("Error occured while training model. CMA-ES error code",
		   solutions.run_status());
      return ModelType::New();
		       
    }

    // Now we use the weights we found in the optimization to train a model and
    // iterate a couple of times to give an idea of how stable the clustering is
    auto eigWeights = gp.pheno( solutions.best_candidate().get_x_dvec() );
    for ( size_t i = 0; i < weights.size(); ++i ) {
      weights[i] = eigWeights(i);
    }

    tracer.Info("Iterations", std::to_string(solutions.niter()));
    tracer.Info("Weights", eigWeights);
 
    double bestRisk = std::numeric_limits<double>::infinity();
    MatrixType bestCentroids;
    ClusterLabelVectorType bestLabels;
    DistanceType dist(weights.data(), weights.size());
    for ( size_t i = 0; i < m_Params.finalNumberOfClusterings; ++i ) {
      InstanceClusteringType clustering = clusterer.Cluster( bags, dist );
      ClusterLabelVectorType labels = ClusterLabelVectorType::Zero( clustering.NumberOfClusters() );
      double risk = labeler.Label( bags, clustering.clusterBagMap, labels );

      tracer.Debug("Risk", risk);

      if ( risk < bestRisk ) {
	bestRisk = risk;
	bestCentroids = clustering.centroids;
	bestLabels = labels;
      }
    }

    m_TrainError = bestRisk;
    typename ModelType::Pointer model = ModelType::New( bestCentroids, bestLabels, weights );
    model->Build();
    return model;
  }
 
protected:
  ParameterType           m_Params;
  ClustererParameterType  m_ClustererParams;
  LabelerParameterType    m_LabelerParams;
  TracerParameterType     m_TracerParams;
  double                  m_TrainError;
};


#endif
