set( LIBS
  ${CMAES_LIBRARIES}
  )

set( progs
  TrainClusterModel
  PredictClusterModel
)

foreach( prog ${progs} )
  add_executable( ${prog} ${prog}.cxx )
  target_link_libraries( ${prog} ${LIBS} )
  install( TARGETS ${prog} DESTINATION tools )
endforeach( prog )
