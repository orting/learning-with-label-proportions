set( LIBS
  cmaes
  )

set( progs
  TrainClusterModel
  PredictClusterModel
)

if( USE_INTERVAL_LABELS )
add_definitions(-DUSE_INTERVAL_LABELS)
foreach( prog ${progs} )
    set( progname ${prog}Interval )
    add_executable( ${progname} ${prog}.cxx )
    target_link_libraries( ${progname} ${LIBS} )
    install( TARGETS ${progname} DESTINATION tools )
  endforeach( prog )

else( USE_INTERVAL_LABELS )
  foreach( prog ${progs} )
    add_executable( ${prog} ${prog}.cxx )
    target_link_libraries( ${prog} ${LIBS} )
    install( TARGETS ${prog} DESTINATION tools )
  endforeach( prog )

endif( USE_INTERVAL_LABELS )
