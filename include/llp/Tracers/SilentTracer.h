#ifndef __SilentTracer_h
#define __SilentTracer_h

#include <iostream>

class SilentTracer {
public:
  enum Level {
    TRACE,
    DEBUG,
    INFO,
    WARNING,
    ERROR
  };
  
  struct ParameterType {    
    ParameterType(Level level=Level::DEBUG)
      : level( level )
    {}
    Level level;
  };

  SilentTracer(const ParameterType& p=ParameterType())
    : m_Level(p.level)
  {}

  template<typename T>
  void Trace(const std::string& s, const T& x) {
    if (m_Level <= Level::TRACE) {
      write("TRACE", s, x);
    }
  }

  template<typename T>
  void Debug(const std::string& s, const T& x) {
    if (m_Level <= Level::DEBUG) {
      write("DEBUG", s, x);
    }
  }
  
  template<typename T>
  void Info(const std::string& s, const T& x) {
    if (m_Level <= Level::INFO) {
      write("INFO", s, x);
    }
  }

  template<typename T>
  void Warning(const std::string& s, const T& x) {
    if (m_Level <= Level::WARNING) {
      write("WARNING", s, x);
    }
  }
  
  template<typename T>
  void Error(const std::string& s, const T&x) {
    if (m_Level <= Level::ERROR) {
      write("ERROR", s, x);
    }
  }

protected:
  template<typename T>
  void write(const std::string&, const std::string&, const T&) { }

  Level m_Level;
};

#endif
