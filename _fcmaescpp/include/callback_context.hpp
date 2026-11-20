// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

#pragma once

namespace fcmaes {

// Data of the objective callback of the current optimization call. The
// callbacks have no user-data argument, so a language binding sets this
// pointer on the calling thread for the duration of one call. The worker
// threads of an evaluator take the value of the thread that creates them.
// Each call has its own value, so calls from different threads, and nested
// calls, do not share callback state.
inline thread_local const void *callback_context = nullptr;

}  // namespace fcmaes
