#ifndef __InstanceClustering_h
#define __InstanceClustering_h

#include <vector>

/**
   A centroid based clustering of bagged instances in bags, with a mapping from
   centroids to bags.
 */
template< typename MatrixType >
struct InstanceClustering {
  InstanceClustering()
    : centroids()
    , clusterBagMap()
    , clusterMembershipIndices()
  {}

  std::size_t NumberOfClusters() const {
    return clusterBagMap.cols();
  }

  std::size_t NumberOfBags() const {
    return clusterBagMap.rows();
  }

  MatrixType centroids, clusterBagMap;
  std::vector< int > clusterMembershipIndices;
};

#endif
