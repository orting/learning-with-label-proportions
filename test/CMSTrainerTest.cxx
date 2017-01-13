/*
  Test CMSTrainer
 */

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>


#include "gtest/gtest.h"

#include "bd/BaggedDataset.h"

#include "Algorithms/KMeansInstanceClusterer.h"
#include "Algorithms/GreedyBinaryClusterLabeler.h"
#include "Algorithms/Trainers/CMSTrainer.h"
#include "Distances/EarthMoversDistance.h"
#include "Distances/WeightedNxMDistance.h"
#include "Losses/ScalarLosses.h"
#include "Losses/ScalarRisk.h"
#include "Tracers/SilentTracer.h"
#include "Tracers/StdOutTracer.h"


class CMSTrainerTest : public ::testing::Test {
public:
  static const size_t InstanceLabelDim = 1;
  static const size_t BagLabelDim = 1;

  typedef ScalarRisk< L1_ScalarLoss > Risk;
  typedef GreedyBinaryClusterLabeler< Risk, BagLabelDim > LabelerType;

  typedef LabelerType::BaggedDatasetType BaggedDatasetType;

  typedef WeightedNxMDistance< EarthMoversDistance > DistanceType;

  typedef ClusterModel< DistanceType, BaggedDatasetType > ModelType;  

  typedef KMeansInstanceClusterer< BaggedDatasetType, DistanceType > ClustererType;
  //typedef KMeansClusteringParameters ClustererParameterType;

  //typedef SilentTracer TracerType;
  typedef StdOutTracer TracerType;
  
  typedef CMSTrainer<BaggedDatasetType, ClustererType, LabelerType, TracerType> TrainerType;
  
  typedef LabelerType::MatrixType MatrixType;
  typedef LabelerType::BagLabelVectorType BagLabelVectorType;
  typedef LabelerType::ClusterLabelVectorType ClusterLabelVectorType;
  typedef BaggedDatasetType::InstanceLabelVectorType InstanceLabelVectorType;
  typedef BaggedDatasetType::IndexVectorType IndexVectorType;

  
protected:
  virtual void SetUp() {
    std::random_device rd;
    std::mt19937 gen(rd());   
    
    std::uniform_int_distribution<size_t> disBags(10, 20);
    std::uniform_int_distribution<size_t> disBagSize(10, 100);
    std::uniform_int_distribution<size_t> disDimension(2, 20);
    std::uniform_real_distribution<double> disProportion(0, 1);
    std::normal_distribution<double> disNeg(0,1);
    std::normal_distribution<double> disPos(10,1);
    
    numberOfBags = disBags( gen );
    bagSize = disBagSize( gen );
    dimension = disDimension( gen );
    dimension += dimension % 2;
    size_t numberOfInstances = numberOfBags * bagSize;

    // Generate random proportions for each bag and sample the cooresponding
    // number of positive and negative instances
    MatrixType instances = MatrixType::Zero( numberOfInstances, dimension );
    BagLabelVectorType bagLabels = BagLabelVectorType::Zero( numberOfBags );
    InstanceLabelVectorType instanceLabels = InstanceLabelVectorType::Zero( numberOfInstances );
    IndexVectorType bagMembership = IndexVectorType::Zero( numberOfInstances );
    
    double size = static_cast<double>(bagSize);
    for ( size_t i = 0; i < numberOfBags; ++i ) {
      size_t posInstances = static_cast<size_t>(std::round(disProportion( gen ) * size));
      bagLabels(i) = static_cast<double>(posInstances) / size;

      size_t posStartIdx = i*bagSize;
      size_t posEndIdx = posStartIdx + posInstances;
      size_t negEndIdx = posStartIdx + bagSize;
      for (size_t j = posStartIdx; j < posEndIdx; ++j) {
	bagMembership(j) = i;
	instanceLabels(j) = 1;
	for (size_t d = 0; d < dimension; ++d) {
	  instances(j,d) = disPos( gen );
	}
      }
      for (size_t j = posEndIdx; j < negEndIdx; ++j) {
	bagMembership(j) = i;
	instanceLabels(j) = 0;
	for (size_t d = 0; d < dimension; ++d) {
	  instances(j,d) = disNeg( gen );
	}
      }
    }
    bags = BaggedDatasetType( instances, bagMembership, bagLabels, instanceLabels );
  }

  BaggedDatasetType bags;
  size_t numberOfBags, bagSize, dimension, numberOfInstances;  
};


TEST_F( CMSTrainerTest, RandomBags ) {
  BaggedDatasetType randomBags = BaggedDatasetType::Random( numberOfBags, bagSize, dimension);
  TrainerType trainer;
  size_t dim = randomBags.Dimension() / 2;
  trainer.Train(randomBags, dim);
  ASSERT_GT(trainer.TrainError(), 0) << "Random bags should not be perfectly predicted";
}


TEST_F( CMSTrainerTest, EasyBags ) {
  TrainerType trainer;
  size_t dim = bags.Dimension() / 2;
  trainer.Train(bags, dim);
  ASSERT_EQ(trainer.TrainError(), 0) << "Bags should be perfectly predicted";
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
