//#define USE_RECORDED_DATA
//#define USE_MACHINE_HALL

using OpenCVForUnity;
using OpenCVForUnityExample;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

public class PointCloudDrawer : MonoBehaviour
{
    private const float near_plane = 0.01f;
    private const float far_plane = 200f;
    private const float sphere_size = 0.008f;

    [SerializeField]
    private string directory_to_png;

    private int current_frame = 0;

    private Texture2D _texture2D;
    private Dictionary<long, Matrix4x4> dict =
        new Dictionary<long, Matrix4x4>();

    private List<string> _images;
    private GameObject _backgroundMesh;

    [SerializeField]
    private Material _material;

    [SerializeField]
    private Camera _camera;

    private static CameraIntrinsics GetIntrinsics()
    {
#if USE_MACHINE_HALL
        //return new CameraIntrinsics
        //(
        //    width: 752,
        //    height: 480,
        //    fx: 458.654f,
        //    fy: 457.296f,
        //    cx: 367.215f,
        //    cy: 248.375f
        //);

        return new CameraIntrinsics
        (
            width: 1280,
            height: 720,

            //fx: 1024.45f,
            //fy: 1031.36f,
            //cx: 673.26f,
            //cy: 438.08f

            fx: 1028.0f,
            fy: 1028.0f,
            cx: 640.0f,
            cy: 360.0f
        );
#else 
        return new CameraIntrinsics
        (
            width: 1280,
            height: 720,

            //fx: 1024.45f,
            //fy: 1031.36f,
            //cx: 673.26f,
            //cy: 438.08f

            fx: 1028.0f,
            fy: 1028.0f,
            cx: 640.0f,
            cy: 360.0f
        );
#endif
    }

    private static DistortionCoefficient GetDistortionCoef()
    {
#if USE_MACHINE_HALL
        //return new DistortionCoefficient()
        //{
        //    k1 = -0.28340811f,
        //    k2 = 0.07395907f,
        //    p1 = 0.00019359f,
        //    p2 = 1.76187114e-05f,
        //    k3 = 0
        //};

        return new DistortionCoefficient()
        {
            k1 = 0.0f,
            k2 = 0.0f,
            p1 = 0.0f,
            p2 = 0.0f,
            k3 = 0.0f
        };
#else
        return new DistortionCoefficient()
        {
            k1 = 0.0f,
            k2 = 0.0f,
            p1 = 0.0f,
            p2 = 0.0f,
            k3 = 0.0f
        };
#endif
    }

    Matrix4x4 YAxisInversion =
        Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));

    private GameObject _root;

    private Mat _camMatrix;

    private uint _width;
    private uint _height;

    public class PairOfMatrixAtFrame
    {
        public Matrix4x4 matrix_slam;
        public Matrix4x4 matrix_aruco;
    }

    private Dictionary<int, PairOfMatrixAtFrame> _pairOfMatrices =
        new Dictionary<int, PairOfMatrixAtFrame>();

    private float _markerLength;
    private Mat _ids;
    private List<Mat> _corners;
    private List<Mat> _rejected;
    private Mat _rvecs;
    private Mat _tvecs;
    private DetectorParameters _detectorParams;
    private Dictionary _dictionary;
    private MatOfDouble _distCoeffs;

#if !USE_RECORDED_DATA
    private IntPtr _slamSystem = IntPtr.Zero;
    private IntPtr _vocabFile;
    private float[] _matrix = new float[16];
    private float _scale_factor;
    private bool _has_computed_scale_factor;
    private Matrix4x4 _matrixFromSlamToAruco;
    private bool _has_matrix;
    private bool _hasCreatePoints;
#endif

    void Start()
    {
        var intrin = GetIntrinsics();
        _width = intrin.Width;
        _height = intrin.Height;

#if USE_RECORDED_DATA
        CreatePointCloud();
        RetrivePoses();
#endif
        GetPngs();
        InitAruco();
        ApplyProjectionMatrix();

#if !USE_RECORDED_DATA
        CreateSlamSystem();
        File.Delete(@"C:\Users\sesa455926\Desktop\movieSlam2\poses.txt");
#else
        //var go = GameObject.Find("Cube");

        Matrix4x4 trs = GetSlamToArucoMatrix();
        ARUtils.SetTransformFromMatrix(_root.transform, ref trs);
#endif
    }

    /// By hand result
    //private static Matrix4x4 GetSlamToArucoMatrix()
    //{
    //    var pos = new Vector3(0.1785481f, 0.1992328f, -0.4355855f);
    //    var rot = new Quaternion(0.1899464f, -0.1183734f, 0.03930989f, 0.9738392f);
    //    var scale = new Vector3(0.6121542f, 0.6121542f, 0.6121542f);

    //    var trs = Matrix4x4.TRS(pos, rot, scale);
    //    return trs;
    //}

    /// Computed but with hand picked scale factor and hand picked frame
    //private static Matrix4x4 GetSlamToArucoMatrix()
    //{
    //    var pos = new Vector3(0.178565f, 0.2048237f, -0.4317436f);
    //    var rot = new Quaternion(0.195333f, -0.1246114f, 0.0449519f, 0.9717491f);
    //    var scale = new Vector3(0.6121542f, 0.6121542f, 0.6121542f);

    //    var trs = Matrix4x4.TRS(pos, rot, scale);
    //    return trs;
    //}

    /// Computed with hand picked frame
    //private static Matrix4x4 GetSlamToArucoMatrix()
    //{
    //    var pos = new Vector3(0.1828432f, 0.1996195f, -0.4255983f);
    //    var rot = new Quaternion(0.1953329f, -0.1246114f, 0.04495191f, 0.9717491f);
    //    var scale = new Vector3(0.6558151f, 0.6558151f, 0.6558151f);

    //    var trs = Matrix4x4.TRS(pos, rot, scale);
    //    return trs;
    //}

    private void OnGUI()
    {
        GUI.Label(new UnityEngine.Rect(0, 0, 200, 200), "frame is " + current_frame);
    }

    private void InitAruco()
    {
        var dictionaryId = 10;
        _markerLength = 0.1f;

        _ids = new Mat();
        _corners = new List<Mat>();
        _rejected = new List<Mat>();
        _rvecs = new Mat();
        _tvecs = new Mat();

        _detectorParams = DetectorParameters.create();
        _dictionary = Aruco.getPredefinedDictionary(dictionaryId);

        var intrin = GetIntrinsics();

        _camMatrix = new Mat(3, 3, CvType.CV_64FC1);
        _camMatrix.put(0, 0, intrin.Fx);
        _camMatrix.put(0, 1, 0);
        _camMatrix.put(0, 2, intrin.Cx);
        _camMatrix.put(1, 0, 0);
        _camMatrix.put(1, 1, intrin.Fy);
        _camMatrix.put(1, 2, intrin.Cy);
        _camMatrix.put(2, 0, 0);
        _camMatrix.put(2, 1, 0);
        _camMatrix.put(2, 2, 1.0f);

        var dist = GetDistortionCoef();
        _distCoeffs = new MatOfDouble(dist.k1, dist.k2, dist.p1, dist.p2);
    }

#if !USE_RECORDED_DATA
    private void CreateSlamSystem()
    {
        var vocabFilePath = Path.Combine(Application.streamingAssetsPath, "vocabulary.bin");
        var vocalFilesPathsAsBytes = Encoding.ASCII.GetBytes(vocabFilePath);

        var cameraConfigFile = Path.Combine(Application.streamingAssetsPath, "cameraConfig.yaml");
        var cameraConfigFiles = Encoding.ASCII.GetBytes(cameraConfigFile);

        var handle1 = GCHandle.Alloc(vocalFilesPathsAsBytes);
        var handle2 = GCHandle.Alloc(cameraConfigFiles);

        var isDisplayingWindow = true;

        byte displayWindowAsByte =
            isDisplayingWindow ?
                (byte)1 :
                (byte)0;

        _vocabFile = SlamWrapper.read_vocab_file(ref vocalFilesPathsAsBytes[0]);

        _slamSystem =
            SlamWrapper.create_SLAM_system(
                _vocabFile,
                ref cameraConfigFiles[0],
                displayWindowAsByte
        );

        handle1.Free();
        handle2.Free();
    }
#endif

    private void GetPngs()
    {
        if (!Directory.Exists(directory_to_png))
        {
            var msg = "Directory " + directory_to_png + " does not exist";
            Debug.Log(msg);
            throw new Exception(msg);
        }

        _images = Directory.GetFiles(directory_to_png, "*.png").ToList();
        var realPath = Path.Combine(directory_to_png, "image-");
        var length = realPath.Length;

        Comparison<string> comparison =
            new Comparison<string>
            (
                (a, b) =>
                {
                    var a_p = a.Remove(0, length).Replace(".png", "");
                    var b_p = b.Remove(0, length).Replace(".png", "");
                    var a_nb = Int32.Parse(a_p);
                    var b_nb = Int32.Parse(b_p);
                    return a_nb - b_nb;
                }
            );

        _images.Sort(comparison);

        var intrinsics = GetIntrinsics();
        var distCoefs = GetDistortionCoef();

        int width = (int)intrinsics.Width;
        int height = (int)intrinsics.Height;

        int halfWidth = width / 2;
        int halfHeight = height / 2;

        _texture2D = new Texture2D(width, height, TextureFormat.ARGB32, false);
        _backgroundMesh = GameObject.CreatePrimitive(PrimitiveType.Quad);
        _backgroundMesh.transform.parent = _camera.transform;
        var renderer = _backgroundMesh.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Custom/undistordShader"));

        mat.SetFloat("_Fx", intrinsics.Fx);
        mat.SetFloat("_Fy", intrinsics.Fy);
        mat.SetFloat("_Cx", intrinsics.Cx);
        mat.SetFloat("_Cy", intrinsics.Cy);

        mat.SetFloat("_K1", distCoefs.k1);
        mat.SetFloat("_K2", distCoefs.k2);
        mat.SetFloat("_P1", distCoefs.p1);
        mat.SetFloat("_P2", distCoefs.p2);
        mat.SetFloat("_K3", distCoefs.k3);

        mat.SetFloat("_Width", intrinsics.Width);
        mat.SetFloat("_Height", intrinsics.Height);

        renderer.material = mat;
        _backgroundMesh.GetComponent<Renderer>().material.mainTexture = _texture2D;

        var aspect_ratio = (float)width / height;

        var frustum =
            BuildFrustumFromIntrinsics
            (
                intrinsics,
                near_plane,
                far_plane
            );

        _backgroundMesh.transform.localScale =
            new Vector3
            (
                x: (frustum.Top - frustum.Bottom) * aspect_ratio,
                y: (frustum.Top - frustum.Bottom),
                z: 1.0f
            );

        var frustumSizeHorizontal = frustum.Right - frustum.Left;
        var frustumSizeVertical = frustum.Top - frustum.Bottom;

        var deltaXPercent = (intrinsics.Cx - halfWidth) / width;
        var deltaYPercent = (intrinsics.Cy - halfHeight) / height;

        _backgroundMesh.transform.localPosition =
            new Vector3
            (
                deltaXPercent * frustumSizeHorizontal,
                deltaYPercent * frustumSizeVertical,
                near_plane + 0.00001f
            );
    }

    private void CreatePointCloud()
    {
        var mapPointsFilePath =
            Path.Combine(Application.streamingAssetsPath, "mappoints.txt");

        var lines = File.ReadAllLines(mapPointsFilePath);

        var pos =
            lines
            .Where(str => str.Contains(' '))
            .Select(str => ToVector3(str))
            .ToArray();

        CreatePointCloudFromVector3dArrayWithReferenceSwitch(pos);
    }

    private void CreatePointCloudFromVector3dArrayWithReferenceSwitch(Vector3[] positions)
    {
        _root = new GameObject();
        _root.transform.position = Vector3.zero;
        _root.transform.rotation = Quaternion.identity;
        _root.name = "Point cloud";

        int i = 0;
        foreach (var pos in positions)
        {
            var obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            obj.transform.parent = _root.transform;
            var m = Matrix4x4.TRS(pos, Quaternion.identity, Vector3.one);
            var new_mat = ChangeReferenceOfMatrix(m);
            var pos_4 = new_mat.GetColumn(3);
            var new_pos = new Vector3(pos_4.x, pos_4.y, pos_4.z);
            obj.transform.position = new_pos;
            obj.transform.localScale = Vector3.one * sphere_size;
            obj.name = "point_" + i.ToString("00000");
            i++;
        }
    }

    private void CreatePointCloudFromVector3dArray(Vector3[] positions)
    {
        _root = new GameObject();
        _root.transform.position = Vector3.zero;
        _root.transform.rotation = Quaternion.identity;
        _root.name = "Point cloud";

        int i = 0;
        foreach (var pos in positions)
        {
            var obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            obj.transform.parent = _root.transform;
            obj.transform.position = pos;
            obj.transform.localScale = Vector3.one * sphere_size;
            obj.name = "point_" + i.ToString("00000");
            i++;
        }
    }

    private Vector3 ToVector3(string str)
    {
        var splits = str.Split(' ');
        return new Vector3(
                x: float.Parse(splits[0]),
                y: float.Parse(splits[1]),
                z: float.Parse(splits[2])
            );
    }

    private void RetrivePoses()
    {
        var poseFilePath =
            Path.Combine(Application.streamingAssetsPath, "poses.txt");

        var lines = File.ReadAllLines(poseFilePath);
        long current_pose = 0;

        for (int i = 0; i < lines.Length;)
        {
            var current_line = lines[i];
            if (current_line.Contains("pose is"))
            {
                current_pose = Int64.Parse(current_line.Split(' ')[0]);
                i++;
            }
            else if (current_line.Contains("[]"))
            {
                dict.Add(current_pose, Matrix4x4.zero);
                i++;
            }
            else if (current_line.Contains("["))
            {
                var line0 = lines[i + 0].Replace("[", "").Replace("]", "").Replace(" ", "").Replace(";", "");
                var line1 = lines[i + 1].Replace("[", "").Replace("]", "").Replace(" ", "").Replace(";", "");
                var line2 = lines[i + 2].Replace("[", "").Replace("]", "").Replace(" ", "").Replace(";", "");
                var line3 = lines[i + 3].Replace("[", "").Replace("]", "").Replace(" ", "").Replace(";", "");

                var mat = GetMatrix(line0, line1, line2, line3);
                dict.Add(current_pose, mat);
#if USE_MACHINE_HALL
                i += 4;
#else
                i += 5;
#endif
            }
            else
            {
                throw new Exception("something went wrong for line " + i + " ie " + current_line);
            }
        }
    }

    private Matrix4x4 GetMatrix
    (
        string line1,
        string line2,
        string line3,
        string line4
    )
    {
#if USE_MACHINE_HALL
        var numbersLine1 = line1.Split(',').Select(str => float.Parse(str)).ToArray();
        var numbersLine2 = line2.Split(',').Select(str => float.Parse(str)).ToArray();
        var numbersLine3 = line3.Split(',').Select(str => float.Parse(str)).ToArray();
        var numbersLine4 = line4.Split(',').Select(str => float.Parse(str)).ToArray();
#else
        var numbersLine1 = line1.Split('\t').Select(str => float.Parse(str)).ToArray();
        var numbersLine2 = line2.Split('\t').Select(str => float.Parse(str)).ToArray();
        var numbersLine3 = line3.Split('\t').Select(str => float.Parse(str)).ToArray();
        var numbersLine4 = line4.Split('\t').Select(str => float.Parse(str)).ToArray();
#endif

        var mat = new Matrix4x4();

        mat[0] = numbersLine1[0];
        mat[1] = numbersLine1[1];
        mat[2] = numbersLine1[2];
        mat[3] = numbersLine1[3];

        mat[4] = numbersLine2[0];
        mat[5] = numbersLine2[1];
        mat[6] = numbersLine2[2];
        mat[7] = numbersLine2[3];

        mat[8] = numbersLine3[0];
        mat[9] = numbersLine3[1];
        mat[10] = numbersLine3[2];
        mat[11] = numbersLine3[3];

        mat[12] = numbersLine4[0];
        mat[13] = numbersLine4[1];
        mat[14] = numbersLine4[2];
        mat[15] = numbersLine4[3];

        return mat.transpose;
    }

    public Matrix4x4 ChangeReferenceOfMatrix(Matrix4x4 m)
    {
        var a = YAxisInversion;
        var b = a.inverse;
        return b * m * a;
    }

    private void OnDestroy()
    {
#if !USE_RECORDED_DATA
        if (_slamSystem.ToInt64() != 0)
        {
            SlamWrapper.shutdown_slam_system(_slamSystem);
            SlamWrapper.delete_pointer(_slamSystem);
            _slamSystem = IntPtr.Zero;
        }

        if (_vocabFile.ToInt64() != 0)
        {
            SlamWrapper.delete_pointer(_vocabFile);
            _vocabFile = IntPtr.Zero;
        }

#endif
    }

    void Update()
    {
        Matrix4x4 maybe_slam_matrix = Matrix4x4.identity;
        bool is_slam_matrix_set = false;

        var stopwatchTex = new System.Diagnostics.Stopwatch();
        stopwatchTex.Start();
        _texture2D.LoadImage(File.ReadAllBytes(_images[current_frame]));
        _texture2D.Apply();
        stopwatchTex.Stop();
        //Debug.LogWarning("elapsed time for texture reading " + stopwatchTex.ElapsedMilliseconds + " ms");

        Mat rgbMat = new Mat((int)_height, (int)_width, CvType.CV_8UC3);
        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Utils.texture2DToMat(_texture2D, rgbMat);
        //Debug.LogWarning("elapsed time for texture conversion is " + stopwatch.ElapsedMilliseconds + " ms");

        Matrix4x4 aruco_mat = Matrix4x4.identity;
        var has_detected = false;

        var thread = new System.Threading.Thread(() =>
        {
            var stopwatchAruco = new System.Diagnostics.Stopwatch();
            stopwatchAruco.Start();
            has_detected = RunAruco(rgbMat, ref aruco_mat);
            stopwatchAruco.Stop();
            //Debug.LogWarning("elapsed time for running aruco is " + stopwatchAruco.ElapsedMilliseconds + " ms");

        });

        thread.Start();

#if !USE_RECORDED_DATA
        var current_index = current_frame;
        var stopwatchSlam = new System.Diagnostics.Stopwatch();
        stopwatchSlam.Start();
        RunSlam(ref maybe_slam_matrix, ref is_slam_matrix_set, rgbMat);
        stopwatchSlam.Stop();
        //Debug.LogWarning("elapsed time for running slam is " + stopwatchSlam.ElapsedMilliseconds + " ms");

#else

#if USE_MACHINE_HALL
            var current_index = dict.Keys.ToArray()[current_frame];
#else
            var current_index = current_frame;
#endif
            if (dict.ContainsKey(current_index))
            {
                maybe_slam_matrix = dict[current_index];
                is_slam_matrix_set = true;
            }
#endif
        thread.Join();
        var slam_matrix = Matrix4x4.identity;

        if (is_slam_matrix_set)
        {
            slam_matrix = ToUnityFrame(maybe_slam_matrix);
            if (!_has_computed_scale_factor)
            {
                _has_computed_scale_factor = TryComputeScaleFactor(out _scale_factor);
                if (_has_computed_scale_factor)
                {
                    Debug.LogWarning("scale factor computed to " + _scale_factor);

                }
            }

            if (_has_matrix)
            {
                if (!_hasCreatePoints && _has_matrix)
                {
                    int array_size = 0;

                    var array_ptr =
                        SlamWrapper.get_3d_tracked_points(_slamSystem, ref array_size);

                    if (array_ptr.ToInt64() != 0 && array_size > 0)
                    {
                        var tracked3DPoints = new float[array_size];
                        Marshal.Copy(array_ptr, tracked3DPoints, 0, array_size);
                        SlamWrapper.delete_pointer(array_ptr);

                        var points = new List<Vector3>();
                        for (int i = 0; i < tracked3DPoints.Length / 4; i++)
                        {
                            var x = tracked3DPoints[i * 4 + 0];
                            var y = tracked3DPoints[i * 4 + 1];
                            var z = tracked3DPoints[i * 4 + 2];
                            var isTracked = tracked3DPoints[i * 4 + 3] != 0 ? true : false;

                            var p = new Vector4(x, y, z, 1);

                            var m = Matrix4x4.TRS(p, Quaternion.identity, Vector3.one);
                            var mat = ChangeReferenceOfMatrix(m);
                            var pos_ = mat.GetColumn(3);
                            pos_ = pos_ / pos_.w;

                            var pos = _matrixFromSlamToAruco * pos_;

                            pos = pos / pos.w;
                            points.Add(pos);

                        }

                        Debug.LogWarning("creating point cloud");
                        CreatePointCloudFromVector3dArray(points.ToArray());
                        _hasCreatePoints = true;
                    }
                }

                // transform camera pose from Slam reference frame
                // to the one in the Aruco one
                var result = _matrixFromSlamToAruco * slam_matrix;
                ApplyPoseMatrix(result);
            }
        }

        if (has_detected)
        {
            //ApplyPoseMatrix(aruco_matrix);

            if (!_pairOfMatrices.ContainsKey(current_index))
            {
                _pairOfMatrices.Add(current_index, new PairOfMatrixAtFrame()
                {
                    matrix_aruco = aruco_mat,
                    matrix_slam = slam_matrix
                });
            }

            // pick good frame to compute relative matrix;
            //if (current_frame == 423)
            //{
            //    // TODO
            //    //var scaleFactor = 0.6121542f;
            //    
            //    var slam_matrix = ToUnityFrame(dict[current_index]);
            //    var p = ComputeSlamToArucoMatrix(slam_matrix, aruco_matrix, scaleFactor);
            //}

            if (is_slam_matrix_set && _has_computed_scale_factor && !_has_matrix)
            {
                //var scaleFactor = 0.6558151126f;
                var scaleFactor = _scale_factor;

                _matrixFromSlamToAruco =
                    ComputeSlamToArucoMatrix(slam_matrix, aruco_mat, scaleFactor);

                Debug.LogWarning("matrix from slam to aruco computed.");

                _has_matrix = true;
            }
        }

        if (Time.frameCount % 1 == 0)
        {
            current_frame++;
#if !USE_RECORDED_DATA
            current_frame = current_frame % _images.Count;
#else
#if USE_MACHINE_HALL
            current_frame = current_frame % dict.Keys.Count;
#else
            current_frame = current_frame % (int) dict.Keys.Max();
            if (current_frame == 0)
            {
                current_frame = 350;
            }
#endif
#endif
        }
    }

    private bool RunAruco(Mat rgbMat, ref Matrix4x4 aruco_matrix)
    {
        var hasDetected = ArucoDetection(rgbMat, out aruco_matrix);
        return hasDetected;
    }

    private void RunSlam(ref Matrix4x4 maybe_slam_matrix, ref bool is_slam_matrix_set, Mat rgbMat)
    {
        IntPtr pose =
            SlamWrapper.update_image
            (
                _slamSystem,
                new IntPtr(rgbMat.dataAddr()),
                (int)_width,
                (int)_height,
                Time.timeSinceLevelLoad
            );

        if (pose.ToInt64() != 0)
        {
            Marshal.Copy(pose, _matrix, 0, 16);
            SlamWrapper.delete_pointer(pose);

            for (int i = 0; i < 16; i++)
            {
                maybe_slam_matrix[i] = _matrix[i];
            }

            is_slam_matrix_set = true;
            maybe_slam_matrix = maybe_slam_matrix.transpose;

            //File.AppendAllText(@"C:\Users\sesa455926\Desktop\movieSlam2\poses.txt",
            //        "" + current_frame + " pose is \n[" + maybe_slam_matrix.ToString() + "]\n"
            //    );

            //if (current_frame == 665)
            //{
            //    int array_size = 0;
            //    var array_ptr =
            //        SlamWrapper.get_3d_tracked_points(_slamSystem, ref array_size);

            //    if (array_ptr.ToInt64() != 0 && array_size > 0)
            //    {
            //        var tracked3DPoints = new float[array_size];
            //        Marshal.Copy(array_ptr, tracked3DPoints, 0, array_size);
            //        SlamWrapper.delete_pointer(array_ptr);

            //        var points = new List<Vector3>();
            //        for (int i = 0; i < tracked3DPoints.Length / 4; i++)
            //        {
            //            var x = tracked3DPoints[i * 4 + 0];
            //            var y = tracked3DPoints[i * 4 + 1];
            //            var z = tracked3DPoints[i * 4 + 2];
            //            var isTracked = tracked3DPoints[i * 4 + 3] != 0 ? true : false;
            //            points.Add(new Vector3(x, y, z));
            //        }

            //        var lines =
            //            points
            //            .Select((point) => "" + point.x + " " + point.y + " " + point.z)
            //            .ToArray();

            //        var path = @"C:\Users\sesa455926\Desktop\movieSlam2\mappoints.txt";
            //        File.WriteAllLines(path, lines);
            //    }
            //}
        }
    }

    private Matrix4x4 ComputeSlamToArucoMatrix
    (
        Matrix4x4 mat_slam,
        Matrix4x4 mat_aruco,
        float scaleFactor
    )
    {

        var fix_scale =
            Matrix4x4.TRS
            (
                Vector3.zero,
                Quaternion.identity,
                Vector3.one * scaleFactor
            );

        var p = mat_aruco * fix_scale * mat_slam.inverse;

        // Extract new local position
        Vector3 position = p.GetColumn(3);

        // Extract new local rotation
        Quaternion rotation = Quaternion.LookRotation(
            p.GetColumn(2),
            p.GetColumn(1)
        );

        // Extract new local scale
        Vector3 scale = new Vector3(
            p.GetColumn(0).magnitude,
            p.GetColumn(1).magnitude,
            p.GetColumn(2).magnitude
        );

        //Debug.LogWarning(position.x + " " + position.y + " " + position.z);
        //Debug.LogWarning(rotation.x + " " + rotation.y + " " + rotation.z + " " + rotation.w);
        //Debug.LogWarning(scale.x + " " + scale.y + " " + scale.z);
        return p;
    }

    private bool TryComputeScaleFactor(out float scale_factor)
    {
        Bounds bb_slam = new Bounds();
        bool is_bb_slam_set = false;
        Bounds bb_aruco = new Bounds();
        bool is_bb_aruco_set = false;
        int max = _pairOfMatrices.Keys.Max();
        var number_of_handled_points = 0;
        for (int i = 0; i < max; i++)
        {
            if
            (
                _pairOfMatrices.ContainsKey(i) // &&
                                               //_pairOfMatrices.ContainsKey(i + 1)
            )
            {
                number_of_handled_points++;
                var matrices_i = _pairOfMatrices[i];

                var aruco_pos_i =
                    new Vector3
                    (
                        matrices_i.matrix_aruco.m03,
                        matrices_i.matrix_aruco.m13,
                        matrices_i.matrix_aruco.m23
                    );

                var slam_pos_i =
                    new Vector3
                    (
                        matrices_i.matrix_slam.m03,
                        matrices_i.matrix_slam.m13,
                        matrices_i.matrix_slam.m23
                    );

                //var matrices_i_plus_one = _pairOfMatrices[i + 1];

                //var aruco_pos_i_one =
                //    new Vector3
                //    (
                //        matrices_i_plus_one.matrix_aruco.m03,
                //        matrices_i_plus_one.matrix_aruco.m13,
                //        matrices_i_plus_one.matrix_aruco.m23
                //    );


                //var slam_pos_i_one =
                //    new Vector3
                //    (
                //        matrices_i_plus_one.matrix_slam.m03,
                //        matrices_i_plus_one.matrix_slam.m13,
                //        matrices_i_plus_one.matrix_slam.m23
                //    );

                //var delta_slam = (slam_pos_i_one - slam_pos_i);
                //var delta_aruco = (aruco_pos_i_one - aruco_pos_i);

                if (!is_bb_slam_set)
                {
                    bb_slam = new Bounds(slam_pos_i, Vector3.zero);
                    is_bb_slam_set = true;
                }
                else
                {
                    bb_slam.Encapsulate(slam_pos_i);
                }

                if (!is_bb_aruco_set)
                {
                    bb_aruco = new Bounds(aruco_pos_i, Vector3.zero);
                    is_bb_aruco_set = true;
                }
                else
                {
                    bb_aruco.Encapsulate(aruco_pos_i);
                }
            }
        }

        if (number_of_handled_points > 50)
        {
            var aruco_magnitude = bb_aruco.extents.magnitude;
            var slam_magnitude = bb_slam.extents.magnitude;

            if (slam_magnitude > 0.3f) // move to at least 0.3m to get a correct estimation
            {
                float average = aruco_magnitude / slam_magnitude;
                scale_factor = average;
                return true;
            }
        }
        scale_factor = 1;
        return false;

    }

    private bool ArucoDetection(Mat rgbMat, out Matrix4x4 mat)
    {
        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Aruco.detectMarkers(rgbMat, _dictionary, _corners, _ids, _detectorParams, _rejected);
        //Debug.LogWarning("elapsed time for marker detection is " + stopwatch.ElapsedMilliseconds + " ms");
        if (_ids.total() > 0)
        {
            stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();
            Aruco.estimatePoseSingleMarkers(_corners, _markerLength, _camMatrix, _distCoeffs, _rvecs, _tvecs);
            //Debug.LogWarning("elapsed time for marker pose retrieval is " + stopwatch.ElapsedMilliseconds + " ms");
            var mat2 = GetMatrix(_tvecs.get(0, 0), _rvecs.get(0, 0));
            mat = mat2;
            Aruco.drawAxis(rgbMat, _camMatrix, _distCoeffs, _rvecs, _tvecs, _markerLength * 0.5f);
            return true;
        }
        mat = Matrix4x4.identity;
        return false;
    }

    private Matrix4x4 GetMatrix(double[] tvec, double[] rv)
    {
        Mat rotMat = new Mat(3, 3, CvType.CV_64FC1);
        Mat rvec = new Mat(3, 1, CvType.CV_64FC1);
        rvec.put(0, 0, rv[0]);
        rvec.put(1, 0, rv[1]);
        rvec.put(2, 0, rv[2]);
        Calib3d.Rodrigues(rvec, rotMat);

        Matrix4x4 transformationM = new Matrix4x4(); // from OpenCV
        transformationM.SetRow(0, new Vector4((float)rotMat.get(0, 0)[0], (float)rotMat.get(0, 1)[0], (float)rotMat.get(0, 2)[0], (float)tvec[0]));
        transformationM.SetRow(1, new Vector4((float)rotMat.get(1, 0)[0], (float)rotMat.get(1, 1)[0], (float)rotMat.get(1, 2)[0], (float)tvec[1]));
        transformationM.SetRow(2, new Vector4((float)rotMat.get(2, 0)[0], (float)rotMat.get(2, 1)[0], (float)rotMat.get(2, 2)[0], (float)tvec[2]));
        transformationM.SetRow(3, new Vector4(0, 0, 0, 1));
        Matrix4x4 invertZM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, 1, -1));
        Matrix4x4 invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));
        Matrix4x4 ARM = invertYM * transformationM;
        ARM = ARM * invertZM;
        return ARM.inverse;
    }

    private Matrix4x4 ToUnityFrame(Matrix4x4 m)
    {
        return ChangeReferenceOfMatrix(m).inverse;
    }

    private void ApplyProjectionMatrix()
    {
        CameraIntrinsics intrinsics = GetIntrinsics();
        var mat = BuildProjMatrixFromIntrinsics(intrinsics, near_plane, far_plane);
        _camera.projectionMatrix = mat;
    }

    private void ApplyPoseMatrix(Matrix4x4 m)
    {
        Vector3 t = m.GetColumn(3);

        Quaternion q =
            Quaternion.LookRotation(
                 m.GetColumn(2),
                 m.GetColumn(1)
            );

        _camera.transform.position = t;
        _camera.transform.rotation = q;
    }

    public static Matrix4x4 BuildProjMatrixFromIntrinsics
    (
        CameraIntrinsics intrinsics,
        float near,
        float far
    )
    {
        Matrix4x4 mat = Matrix4x4.zero;
        mat.m00 = 2.0f * intrinsics.Fx / intrinsics.Width;
        mat.m11 = 2.0f * intrinsics.Fy / intrinsics.Height;

        mat.m02 = 2.0f * (intrinsics.Cx / intrinsics.Width) - 1.0f;
        mat.m12 = 2.0f * (intrinsics.Cy / intrinsics.Height) - 1.0f;
        mat.m22 = -(far + near) / (far - near);
        mat.m32 = -1;

        mat.m23 = 2.0f * far * near / (near - far);
        return mat;
    }

    public struct Frustum
    {
        public float Left;
        public float Right;
        public float Bottom;
        public float Top;
        public float Near;
        public float Far;
    }

    public static Frustum BuildFrustumFromIntrinsics
    (
        CameraIntrinsics intrinsics,
        float near,
        float far
    )
    {
        float left = -intrinsics.Cx / intrinsics.Fx;
        float right = (intrinsics.Width - intrinsics.Cx) / intrinsics.Fx;
        float top = intrinsics.Cy / intrinsics.Fy;
        float bottom = -(intrinsics.Height - intrinsics.Cy) / intrinsics.Fy;

        return new Frustum()
        {
            Left = left * near,
            Right = right * near,
            Top = top * near,
            Bottom = bottom * near,
            Near = near,
            Far = far
        };
    }

    public struct CameraIntrinsics
    {
        public float Fx;
        public float Fy;
        public float Cx;
        public float Cy;

        public uint Width;
        public uint Height;

        public CameraIntrinsics
        (
            uint width,
            uint height,
            float fx,
            float fy,
            float cx,
            float cy
        )
        {
            Width = width;
            Height = height;
            Fx = fx;
            Fy = fy;
            Cx = cx;
            Cy = cy;
        }
    }

    public class DistortionCoefficient
    {
        public float k1;
        public float k2;
        public float p1;
        public float p2;
        public float k3;
    }
}
