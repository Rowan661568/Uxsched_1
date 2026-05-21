add_hal_lib(rtx4060 LINK_LIBRARIES cuxtra)
add_shim_lib(rtx4060
    SOFTLINKS_LINUX libcuda.so.1 libcuda.so
    TARGET_NAME_WINDOWS "nvcuda"
)
