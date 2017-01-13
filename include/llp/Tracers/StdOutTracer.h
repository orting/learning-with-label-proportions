#ifndef __StdOutTracer_h
#define __StdOutTracer_h

#include <iostream>
#include "SilentTracer.h"

class StdOutTracer : public SilentTracer {
public:   
  StdOutTracer(const ParameterType& params)
    : m_Level(params.level)
  {}

protected:
  template<typename T>
  void write( const std::string& prefix, const std::string& msg, const T& x) {
    std::cout << '[' << prefix << "] " << msg << " : " << x << '\n';
  }
  
  Level m_Level;
};

#endif
