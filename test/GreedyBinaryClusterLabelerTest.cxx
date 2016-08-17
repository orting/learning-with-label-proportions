/*
  Test CMSTrainer
 */

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>


#include "gtest/gtest.h"

#include "Losses/IntervalLosses.h"
#include "Losses/IntervalRisk.h"
#include "Algorithms/GreedyBinaryClusterLabeler.h"



class GreedyBinaryClusterLabelerTest : public ::testing::Test {
public:
  typedef IntervalRisk< L1_IntervalLoss > Risk;
  typedef GreedyBinaryClusterLabeler< Risk, 2 > Labeler;
  typedef Labeler::BaggedDatasetType BaggedDatasetType;
  typedef BaggedDatasetType::IndexVectorType IndexVectorType;
  
  typedef Labeler::MatrixType MatrixType;
  typedef Labeler::BagLabelVectorType BagLabelVectorType;
  typedef Labeler::ClusterLabelVectorType ClusterLabelVectorType;
  
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
  }

};


TEST_F( GreedyBinaryClusterLabelerTest, TwoInstancesTwoBagsTwoClusters ) {
  auto instances = MatrixType::Zero(2,1);
  auto instanceLabels = ClusterLabelVectorType::Zero(2);

  IndexVectorType indices(2);
  indices << 0, 1;  

  BagLabelVectorType bagLabels(2,2);
  bagLabels <<  1, 1, 0, 0;
  
  BaggedDatasetType bags( instances, indices, bagLabels, instanceLabels );
  std::vector< int > clusterIndices{0, 1};
  MatrixType clusterBagMap(2,2);
  clusterBagMap << 1, 0, 0, 1;

  ClusterLabelVectorType clusterLabels(2);

  Labeler labeler;

  ASSERT_EQ( 0, labeler.Label( bags, clusterBagMap, clusterLabels ) );
  ASSERT_EQ( 1, clusterLabels(0) );
  ASSERT_EQ( 0, clusterLabels(1) );    
}


TEST_F( GreedyBinaryClusterLabelerTest, UnitClusterBagMap ) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> disLabel(0, 1);
  size_t nBags = 100;
  auto instances = MatrixType::Zero(nBags,1);
  auto instanceLabels = ClusterLabelVectorType::Zero(nBags);

  auto indices = IndexVectorType::Zero(nBags);

  BagLabelVectorType bagLabels(nBags,2);
  for ( size_t i = 0; i < nBags; ++i ) {
    bagLabels(i,0) = bagLabels(i,1) = disLabel( gen ) ;
  }
    
  BaggedDatasetType bags( instances, indices, bagLabels, instanceLabels );
  auto clusterBagMap = MatrixType::Identity( nBags, nBags );
  ClusterLabelVectorType clusterLabels( nBags );

  Labeler labeler;

  ASSERT_EQ( 0, labeler.Label( bags, clusterBagMap, clusterLabels ) );
  for ( size_t i = 0; i < nBags; ++i ) {
    ASSERT_EQ( bags.BagLabels()(i,0), clusterLabels(i) );
    ASSERT_EQ( bags.BagLabels()(i,1), clusterLabels(i) );
  }
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
