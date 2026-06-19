#pragma once

#include "xsched/types.h"
#include "xsched/ascend/hal/acl.h"

namespace xsched::ascend
{

class AscendMemoryManager
{
public:
    static void RegisterDeviceAllocation(void *ptr, size_t size,
                                         aclrtContext ctx, int32_t dev);
    static void UnregisterAllocation(void *ptr);

    static void OnQueueSuspend(aclrtContext ctx, int32_t dev, XQueueHandle queue);
    static void BeforeQueueResume(aclrtContext ctx, int32_t dev, XQueueHandle queue);
};

} // namespace xsched::ascend
