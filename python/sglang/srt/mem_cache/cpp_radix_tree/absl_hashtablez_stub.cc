#include "absl/container/internal/hashtablez_sampler.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace container_internal {

HashtablezInfoHandle ForcedTrySample(size_t, size_t, size_t, uint16_t) {
  return HashtablezInfoHandle(nullptr);
}

}  // namespace container_internal
ABSL_NAMESPACE_END
}  // namespace absl
