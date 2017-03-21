/* 
   Train a CMS model with binary cluster labeling and bag proportion gold standard
*/

#include <iostream>
#include <fstream>

#include "tclap/CmdLine.h"

#include "bd/BaggedDataset.h"

#include "Algorithms/KMeansInstanceClusterer.h"
#include "Algorithms/GreedyBinaryClusterLabeler.h"
#include "Algorithms/Trainers/CMSTrainer.h"
#include "Algorithms/Trainers/CMSTrainerParameters.h"
#include "Distances/EarthMoversDistance.h"
#include "Distances/WeightedNxMDistance.h"
#include "Losses/ScalarLosses.h"
#include "Losses/ScalarRisk.h"
#include "Losses/IntervalLosses.h"
#include "Losses/IntervalRisk.h"
#include "Tracers/FileTracer.h"
#include "Tracers/StdOutTracer.h"

int main(int argc, char *argv[]) {
  TCLAP::CmdLine cmd("TrainClusterModel", ' ', LLP_VERSION);

  // We need a path to some data
  TCLAP::ValueArg<std::string> 
    baggedDatasetArg("b", 
		     "bags", 
		     "Path to bagged dataset",
		     true,
		     "",
		     "path", 
		     cmd);
  
  TCLAP::ValueArg<size_t> 
    nHistogramsArg("n", 
		   "histograms", 
		   "Number of histograms",
		   true,
		   1,
		   "size_t", 
		   cmd);

  TCLAP::ValueArg<int> 
    branchingArg("B", 
		 "branching", 
		 "Branching parameter to pass to flann",
		 true,
		 2,
		 ">=2", 
		 cmd);

  TCLAP::ValueArg<int> 
    kMeansIterationsArg("I", 
			"kmeans-iterations", 
			"Iterations parameter to pass to flann",
			false,
			11,
			">=2", 
			cmd);

  
  TCLAP::ValueArg<size_t> 
    kArg("k", 
	 "clusters", 
	 "Number of clusters",
	 true,
	 2,
	 ">=2", 
	 cmd);
  
  
  TCLAP::ValueArg<std::string> 
    outputArg("o", 
	      "output", 
	      "Base path for output files",
	      true,
	      "",
	      "path", 
	      cmd);

    TCLAP::ValueArg<int> 
    maxItersArg("m", 
		"max-iterations", 
		"Maximum iterations of CMA-ES. Set to <= 0 to let CMA-ES decide.",
		false,
		0,
		"it", 
		cmd);
  
  try {
    cmd.parse(argc, argv);
  } catch(TCLAP::ArgException &e) {
    std::cerr << "Error : " << e.error() 
	      << " for arg " << e.argId() 
	      << std::endl;
    return EXIT_FAILURE;
  }

  // Store the arguments
  const std::string baggedDatasetPath{ baggedDatasetArg.getValue() };
  const size_t nHistograms{ nHistogramsArg.getValue() };
  const int branching{ branchingArg.getValue() };
  const int kMeansIterations{ kMeansIterationsArg.getValue() };
  const size_t k{ kArg.getValue() };
  const std::string outputPath{ outputArg.getValue() };
  const int maxIters{ maxItersArg.getValue() };  
  //// Commandline parsing is done ////
  
  /* const size_t InstanceLabelDim = 1; */
#ifdef USE_INTERVAL_LABELS
  const size_t BagLabelDim = 2;
  typedef IntervalRisk< L1_IntervalLoss > RiskType;
#else
  const size_t BagLabelDim = 1;
  typedef ScalarRisk< L1_ScalarLoss > RiskType;
#endif
  typedef GreedyBinaryClusterLabeler< RiskType, BagLabelDim > LabelerType;
  typedef typename LabelerType::ParameterType LabelerParameterType;

  typedef LabelerType::BaggedDatasetType BaggedDatasetType;

  typedef WeightedNxMDistance< EarthMoversDistance > DistanceType;

  typedef KMeansInstanceClusterer< BaggedDatasetType, DistanceType > ClustererType;
  typedef typename ClustererType::ParameterType ClustererParameterType;

  typedef FileTracer TracerType;
  typedef typename TracerType::ParameterType TracerParameterType;
  
  typedef CMSTrainer<BaggedDatasetType, ClustererType, LabelerType, TracerType> TrainerType;
  typedef typename TrainerType::ParameterType TrainerParameterType;
  typedef typename TrainerType::ModelType ModelType;  

  std::ifstream baggedDatasetIs( baggedDatasetPath );    
  BaggedDatasetType bags = BaggedDatasetType::LoadText( baggedDatasetIs, true );

  ClustererParameterType clustererParams(k, branching, kMeansIterations);
  LabelerParameterType labelerParams;
  TracerParameterType tracerParams(TracerType::Level::INFO, outputPath + ".cms.trace");  

  std::string cmaTrace(outputPath + ".cma.trace");
  TrainerParameterType trainerParams(
    maxIters,         // Maximum number of iterations of CMA-ES
    cmaTrace,         // Trace file base path
    0.5,              // Sigma for CMA-ES
    -1,               // Lambda for CMA-ES
    0,                // Random seed for CMA-ES
    false,            // Toggle trace for trainer
    10                // Number of clusterings to run after optimization of feature weights is done
  );

  TrainerType trainer( trainerParams, clustererParams, labelerParams, tracerParams );
  ModelType::Pointer model = trainer.Train( bags, nHistograms );
  
  // Save the model
  std::ofstream os( outputPath + ".model");
  os << *model;
  
  return 0;
}
