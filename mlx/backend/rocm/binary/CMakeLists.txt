target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/add.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/arctan2.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bitwise_binary.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/divide.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/equal.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/greater.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/greater_equal.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/less.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/less_equal.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/logical_and.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/logical_or.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/log_add_exp.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/minimum.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/maximum.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/multiply.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/power.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/remainder.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/not_equal.hip.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/subtract.hip.cpp)
