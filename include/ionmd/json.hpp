#ifndef IONMD_JSON_HPP
#define IONMD_JSON_HPP

// Wrapper around the bundled nlohmann::json 2.1.1 that silences warnings the
// old header triggers with modern compilers.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <json.hpp>

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif
