#####
##
## Copyright (C) 2012 by the deal.II authors
##
## This file is part of the deal.II library.
##
## <TODO: Full License information>
## This file is dual licensed under QPL 1.0 and LGPL 2.1 or any later
## version of the LGPL license.
##
## Author: Matthias Maier <matthias.maier@iwr.uni-heidelberg.de>
##
#####

#
# Configuration for the ARPACK library:
#

OPTION(DEAL_II_WITH_ARPACK
  "Build deal.II with support for arpack."
  OFF)


MACRO(FEATURE_ARPACK_FIND_EXTERNAL var)
  FIND_PACKAGE(ARPACK)

  IF(ARPACK_FOUND)
    SET(${var} TRUE)
  ENDIF()
ENDMACRO()


MACRO(FEATURE_ARPACK_CONFIGURE_EXTERNAL var)

  LIST(APPEND DEAL_II_EXTERNAL_LIBRARIES ${ARPACK_LIBRARIES})
  ADD_FLAGS(CMAKE_SHARED_LINKER_FLAGS "${ARPACK_LINKER_FLAGS}")

  SET(DEAL_II_USE_ARPACK TRUE)

  SET(${var} TRUE)
ENDMACRO()


CONFIGURE_FEATURE(ARPACK)

