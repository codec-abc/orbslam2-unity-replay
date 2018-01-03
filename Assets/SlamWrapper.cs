using System;
using System.Runtime.InteropServices;


public static class SlamWrapper
{
    private const string dllName = "orb_slam2_windows.dll";

    [DllImport(dllName)]
    public static extern /*void* */ IntPtr read_vocab_file(
        /*const char* */ ref byte vocabulary_file_path
    );

    [DllImport(dllName)]
    public static extern IntPtr create_SLAM_system(
        IntPtr /* const void* */ vocabulary_file_path,
        ref byte /* const char* */  camera_config_file,
        byte /* unsigned char */ display_window
    );

    [DllImport(dllName)]
    public static extern int get_tracking_state(
        /* void* */ IntPtr slam_system_ptr
    );

    [DllImport(dllName)]
    public static extern void reset_slam_system(
        /* void* */ IntPtr slam_system_ptr
    );

    [DllImport(dllName)]
    public static extern void activate_localization_mode(
        /* void* */ IntPtr slam_system_ptr
    );

    [DllImport(dllName)]
    public static extern void deactivate_localization_mode(
        /* void* */ IntPtr slam_system_ptr
    );

    [DllImport(dllName)]
    public static extern void shutdown_slam_system(
        IntPtr /* void* */ slam_system_ptr
    );

    [DllImport(dllName)]
    public static extern void delete_pointer(IntPtr /* void* */ pointer);


    [DllImport(dllName)]
    public static extern void free_pointer(IntPtr /* void* */ pointer);

    [DllImport(dllName)]
    public static extern IntPtr /* float* */
        update_image(
            /* void* */ IntPtr slam_system_ptr,
            /* unsigned char* */ IntPtr image_data,
            int width,
            int height,
            double time_stamp
        );

    [DllImport(dllName)]
    public static extern IntPtr /* float* */
        get_tracked_screen(
            /* void* */ IntPtr slam_system_ptr,
            /* int* */ ref int array_size
        );

    [DllImport(dllName)]
    public static extern IntPtr /* float* */
        get_3d_tracked_points(
            /* void* */ IntPtr slam_system_ptr,
            /* int* */ ref int array_size
        );
}

