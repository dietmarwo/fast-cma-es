#pragma once

#include "common.hpp"

namespace fcmaes::bindings {

template <class Fn>
auto without_gil(Fn &&fn) -> decltype(fn()) {
    nb::gil_scoped_release release;
    return fn();
}

}  // namespace fcmaes::bindings
