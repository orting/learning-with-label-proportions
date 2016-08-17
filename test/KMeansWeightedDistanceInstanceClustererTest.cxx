/*
  Test KMeansWeightedDistanceInstanceClusterer
 */

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>


#include "gtest/gtest.h"

#include "Algorithms/KMeansWeightedDistanceInstanceClusterer.h"
#include "Distances/WeightedEarthMoversDistance2.h"
#include "bd/BaggedDataset.h"

class KMeansWeightedDistanceInstanceClustererTest : public ::testing::Test {
public:
  typedef WeightedEarthMoversDistance2 DistanceType;
  typedef BaggedDataset< 1, 1 > BaggedDatasetType;

  typedef KMeansWeightedDistanceInstanceClusterer< BaggedDatasetType, DistanceType > ClustererType;
  typedef KMeansClusteringParameters ParameterType;
  
  typedef typename BaggedDatasetType::MatrixType MatrixType;
  typedef typename BaggedDatasetType::InstanceLabelVectorType InstanceLabelVectorType;
  typedef typename BaggedDatasetType::BagLabelVectorType BagLabelVectorType;
  typedef typename BaggedDatasetType::IndexVectorType IndexVectorType;
  
protected:
  virtual void SetUp() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> disSize(3, 50);

    srand((unsigned int) time(0));
    instances = MatrixType::Random(200, 2);
    instances /= instances.maxCoeff();
    instances.topRows(100).array() -= 10;
    instances.bottomRows(100).array() += 10;   
    
    instanceLabels = InstanceLabelVectorType::Zero(200);
    instanceLabels.topRows(100) = InstanceLabelVectorType::Ones(100);

    size_t nBags = 5;
    std::uniform_int_distribution<size_t> disIndex(0,nBags-1);
    bagMembershipIndices = IndexVectorType::Zero(200);
    std::vector<size_t> bagSizes( nBags, 0 );    
    for ( size_t i = 0; i < 200; ++i ) {
      auto idx = disIndex( gen );
      bagMembershipIndices(i) = idx;
      ++bagSizes[idx];      
    }  

    bagLabels = BagLabelVectorType::Zero(nBags);
    for ( size_t i = 0; i < 200; ++i ) {
      if ( instanceLabels(i) == 1 ) {
    	++bagLabels(bagMembershipIndices(i));
      }
    }
    for ( std::size_t i = 0; i < nBags; ++i ) {
      bagLabels(i) /= bagSizes[i];
    }    
  }

  MatrixType instances;
  InstanceLabelVectorType instanceLabels;
  IndexVectorType bagMembershipIndices;
  BagLabelVectorType bagLabels;
};


TEST_F( KMeansWeightedDistanceInstanceClustererTest, TwoClusters ) {
  std::vector< double > weights( instances.cols(), 0.5 );    
  BaggedDatasetType bags( instances, bagMembershipIndices, bagLabels, instanceLabels );
  ParameterType params(2);
  ClustererType c(params);
  auto clustering = c.Cluster(bags, weights.data(), weights.size() );
  
  ASSERT_EQ( 2, clustering.NumberOfClusters() );
  auto expC1 = instances.topRows(100).colwise().mean();
  auto expC2 = instances.bottomRows(100).colwise().mean();
  
  auto c1 = clustering.centroids.row(0);
  auto c2 = clustering.centroids.row(1);
  
  if ( expC1.isApprox(c1) ) {
    ASSERT_TRUE( expC2.isApprox( c2 ) );
  }
  else {
    ASSERT_TRUE( expC1.isApprox( c2 ) );
    ASSERT_TRUE( expC2.isApprox( c1 ) );
  }

  for ( int i = 0; i < clustering.clusterBagMap.rows(); ++i ) {
    ASSERT_EQ( 1.0, clustering.clusterBagMap.row(i).sum() );
  }
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
