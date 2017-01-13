/* 
   Test a CMS model
*/

#include <algorithm>
#include <cmath>
#include <fstream>
#include <unordered_map>

#include "Eigen/Dense"

#include "tclap/CmdLine.h"

#include "bd/BaggedDataset.h"

#include "Distances/EarthMoversDistance.h"
#include "Distances/WeightedNxMDistance.h"
#include "Models/ClusterModel.h"

int main(int argc, char *argv[]) {
  TCLAP::CmdLine cmd("PredictClusterModel", ' ', LLP_VERSION);

  TCLAP::ValueArg<std::string> 
    baggedDatasetArg("b", 
		     "bags", 
		     "Path to bagged dataset",
		     true,
		     "",
		     "path", 
		     cmd);
  

  TCLAP::ValueArg<std::string> 
    modelArg("M", 
	     "model", 
	     "Path to model.",
	     true,
	     "",
	     "path", 
	     cmd);
  

  TCLAP::ValueArg<std::string> 
    outputArg("o", 
	      "output", 
	      "Base path for output files",
	      true,
	      "",
	      "path", 
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
  const std::string modelPath{ modelArg.getValue() };
  const std::string outputPath{ outputArg.getValue() };  
  //// Commandline parsing is done ////
  
  const int BagLabelDim = 1;
  const int InstanceLabelDim = 1;
  typedef BaggedDataset<BagLabelDim, InstanceLabelDim> BaggedDatasetType;
  typedef WeightedNxMDistance< EarthMoversDistance > DistanceType;
  typedef ClusterModel< DistanceType, BaggedDatasetType > ModelType;
  
  BaggedDatasetType bags = BaggedDatasetType::LoadText( baggedDatasetPath, true );
  ModelType::Pointer model = ModelType::Load( modelPath );
  model->Predict(bags);

  std::ofstream os(outputPath);
  os << bags.InstanceLabels();
  
  return EXIT_SUCCESS;
}
