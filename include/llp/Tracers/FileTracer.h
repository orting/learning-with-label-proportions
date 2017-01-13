#ifndef __FileTracer_h
#define __FileTracer_h

#include "SilentTracer.h"

class FileTracer : public SilentTracer {
public:
  struct ParameterType : public SilentTracer::ParameterType {
    ParameterType(Level level=Level::DEBUG, const std::string& path=std::string())
      : SilentTracer::ParameterType(level)
      , path(path)
    {}
    
    std::string path;
  };
    
  FileTracer(const ParameterType& params)
    : SilentTracer(params)
    , m_Out(params.path)
  {}

  
protected:
  template<typename T>
  void write(const std::string& prefix, const std::string& msg, const T& x) {
    m_Out << '[' << prefix << "] " << msg << " : " << x << '\n';
  }
  
  std::ofstream m_Out;
};

#endif
