find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set( LIBS
  gtest
  gtest_main
  pthread
  cmaes
  )

set( progs
  CMSModelTest
  CMSTrainerTest
  CoOccurenceMatrixTest
  GreedyBinaryClusterLabelerTest
  InstanceClusteringTest
  IntervalLossesTest
  IntervalRiskTest
  KMeansWeightedDistanceInstanceClustererTest
  RandomMatrixTest
  WeightedNxMDistanceTest
  )

foreach( prog ${progs} )
  add_executable( ${prog} ${prog}.cxx )
  target_link_libraries( ${prog} ${LIBS} )
  add_test( NAME ${prog} COMMAND ${prog} --gtest_repeat=1 )
endforeach( prog )


# project( EE_Test )

# set( GTEST_LIBRARIES
#   gtest
#   gtest_main
#   )

# set( LIBS
#   ${ITK_LIBRARIES}
#   ${GTEST_LIBRARIES}
#   ${CERES_LIBRARIES}
#   )

# set( progs
#   BaggedDatasetTest
#   CMSModelTest
#   CMSTrainerTest
#   DenseHistogramTest
#   EigenvaluesTest
#   GreedyBinaryClusterLabelerTest
#   HausdorffTest
#   IntervalLossesTest
#   IntervalRiskTest
#   WeightedEarthMoversDistanceTest
#   )

# foreach( prog ${progs} )
#   add_executable( ${prog} ${prog}.cxx )
#   target_link_libraries( ${prog} ${LIBS} )
#   install( TARGETS ${prog} DESTINATION bin )
#   add_test( NAME ${prog} COMMAND ${prog} --gtest_repeat=1 )
# endforeach( prog )

# # add_custom_command(
# #   TARGET ClusterModelTest POST_BUILD
# #   COMMAND ${CMAKE_COMMAND} -E copy_directory
# #   ${CMAKE_SOURCE_DIR}/Test/Data $<TARGET_FILE_DIR:ClusterModelTest>/Data)


# #  ClusterLabellerTest
# #  ClusterModelTest
# #  CrossValidatorTest
# #  
# #  
