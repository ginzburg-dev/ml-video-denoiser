include(GoogleTest)

add_executable(denoiser_tests
    test_tensor.cc
    test_weight_loader.cc
    test_conv2d.cc
    test_batchnorm2d.cc
    test_unet_forward.cc
)

target_link_libraries(denoiser_tests PRIVATE
    denoiser_engine
    GTest::gtest_main
)

target_include_directories(denoiser_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

gtest_discover_tests(denoiser_tests)
