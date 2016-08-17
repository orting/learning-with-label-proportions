/*
  Test InstanceClustering
 */

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>

#include "Eigen/Dense"

#include "gtest/gtest.h"

#include "Algorithms/InstanceClustering.h"


class InstanceClusteringTest : public ::testing::Test {
public:
  typedef Eigen::Matrix< double,
			 Eigen::Dynamic,
			 Eigen::Dynamic,
			 Eigen::RowMajor > MatrixType;
  typedef InstanceClustering< MatrixType > ClusteringType;



  
protected:
  InstanceClusteringTest()
    : numberOfClusters()
    , numberOfBags()
    , clustering()
  {}
  
  virtual void SetUp() {
    numberOfClusters = 5;
    numberOfBags = 10;
    clustering.clusterBagMap = MatrixType::Random(numberOfBags, numberOfClusters);
  }

  size_t numberOfClusters, numberOfBags;
  ClusteringType clustering;

};


TEST_F( InstanceClusteringTest, NumberOfClusters ) {
  ASSERT_EQ( numberOfClusters, clustering.NumberOfClusters() );
}


TEST_F( InstanceClusteringTest, NumberOfBags ) {
  ASSERT_EQ( numberOfBags, clustering.NumberOfBags() );
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
