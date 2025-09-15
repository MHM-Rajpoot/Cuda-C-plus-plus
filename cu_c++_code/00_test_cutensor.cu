#include <cutensor.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    cutensorHandle_t handle;
    cutensorStatus_t status = cutensorCreate(&handle);

    if (status == CUTENSOR_STATUS_SUCCESS) {
        std::cout << "✅ cuTENSOR initialized successfully!" << std::endl;
    } else {
        std::cout << "❌ cuTENSOR init failed: " << status << std::endl;
    }

    cutensorDestroy(handle); // cleanup
    return 0;
}
